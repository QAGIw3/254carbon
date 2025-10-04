"""
Base connector SDK for data ingestion.

Overview
--------
Defines the ``Ingestor`` abstract base class that all data source connectors
must implement. A connector is responsible for:
- Discovering available streams/collections at the source
- Pulling or subscribing to source data
- Mapping records to a canonical schema
- Emitting records to downstream systems (e.g., Kafka)
- Persisting checkpoints for reliable resumption

Safety & Operations
-------------------
- Checkpointing is rate‑limited to reduce DB churn.
- Database errors in checkpointing are surfaced, but the outer run() tries to
  persist error state and always closes the pool.
- The DB schema is created lazily on startup for convenience; in production,
  manage this via migrations.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
import asyncio
from datetime import datetime, timezone
import logging
import json
import time

import asyncpg

logger = logging.getLogger(__name__)


class Ingestor(ABC):
    """
    Base class for all data source connectors.
    
    Connectors are plugins that pull data from external sources,
    map to canonical schema, and emit to Kafka.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize connector with configuration.

        Args:
            config: Connector-specific configuration dict
        """
        self.config = config
        self.source_id = config.get("source_id")
        self.checkpoint_state: Dict[str, Any] = {}

        # PostgreSQL connection configuration for checkpoints
        self.db_config = config.get("database", {})
        self.db_host = self.db_config.get("host", "postgresql")
        self.db_port = self.db_config.get("port", 5432)
        self.db_name = self.db_config.get("database", "market_intelligence")
        self.db_user = self.db_config.get("user", "postgres")
        self.db_password = self.db_config.get("password", "postgres")
        self.db_app_name = self.db_config.get("application_name", f"connector_{self.source_id}")
        # Throttle checkpoint writes to avoid excessive DB load
        self.checkpoint_min_interval_seconds = self.db_config.get("checkpoint_min_interval_seconds", 30)
        self._last_checkpoint_write_ms: Optional[int] = None

        # Async event loop dedicated to checkpoint IO
        self._loop = asyncio.new_event_loop()
        self._db_pool: Optional[asyncpg.pool.Pool] = None

        # Initialize database connection (creates tables if missing)
        self._loop.run_until_complete(self._init_db())
    
    @abstractmethod
    def discover(self) -> Dict[str, Any]:
        """
        Discover available data streams from the source.
        
        Returns:
            Dict with metadata about available streams:
            {
                "streams": [
                    {"name": "nodal_lmp", "fields": [...], "update_freq": "5min"},
                    ...
                ]
            }
        """
        pass
    
    @abstractmethod
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull or subscribe to data from source.
        
        For batch sources: pull data since last checkpoint.
        For streaming sources: subscribe and yield messages.
        
        Yields:
            Raw message dicts from the source
        """
        pass
    
    @abstractmethod
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map raw source data to canonical schema.
        
        Args:
            raw: Raw message from source
        
        Returns:
            Message dict conforming to canonical Avro schema
        """
        pass
    
    @abstractmethod
    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """
        Emit events to Kafka topic.
        
        Args:
            events: Iterator of canonical schema messages
        
        Returns:
            Number of events emitted
        """
        pass
    
    @abstractmethod
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """
        Save checkpoint state for resume/recovery with retry/backoff and history logging.

        Args:
            state: State dict to persist (e.g., last_timestamp, last_offset)
        """
        # Respect min interval between writes to reduce DB churn
        now_ms = int(time.time() * 1000)
        if self._last_checkpoint_write_ms is not None:
            if (now_ms - self._last_checkpoint_write_ms) < (self.checkpoint_min_interval_seconds * 1000):
                return

        self._loop.run_until_complete(self._checkpoint_async(state))
        self._last_checkpoint_write_ms = now_ms
        logger.debug(f"Checkpoint saved for {self.source_id}: {state}")
    
    def validate_event(self, event: Dict[str, Any]) -> bool:
        """
        Validate event against data quality rules.

        Args:
            event: Canonical schema event

        Returns:
            True if valid, False otherwise
        """
        # Basic validation (extend per-connector if stricter rules are needed)
        required_fields = [
            "event_time_utc",
            "market",
            "product",
            "instrument_id",
            "value",
        ]

        for field in required_fields:
            if field not in event:
                logger.warning(f"Missing required field: {field}")
                return False

        # Value range check
        value = event.get("value")
        if value is not None and (value < -1000 or value > 100000):
            logger.warning(f"Value out of range: {value}")
            return False

        return True

    async def _init_db(self) -> None:
        """Initialize database connection and create checkpoint tables if needed."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS connector_checkpoints (
                        connector_id VARCHAR(255) PRIMARY KEY,
                        last_event_time TIMESTAMP WITH TIME ZONE,
                        last_successful_run TIMESTAMP WITH TIME ZONE,
                        state JSONB,
                        error_count INTEGER DEFAULT 0,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS connector_checkpoint_history (
                        history_id BIGSERIAL PRIMARY KEY,
                        connector_id VARCHAR(255) NOT NULL,
                        state JSONB,
                        metadata JSONB,
                        status VARCHAR(20),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_connector_checkpoint_history__connector_time
                        ON connector_checkpoint_history (connector_id, created_at DESC);
                    """
                )

            logger.info(f"Database initialized for connector {self.source_id}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load last checkpoint state from PostgreSQL.

        Returns:
            Last checkpoint state dict or None
        """
        try:
            result = self._loop.run_until_complete(self._load_checkpoint_async())
            if result:
                self.checkpoint_state = result
                logger.info(f"Loaded checkpoint for {self.source_id}: {self.checkpoint_state}")
                return self.checkpoint_state
            logger.info(f"No checkpoint found for {self.source_id}")
            return None

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def run(self) -> int:
        """
        Main execution loop: pull, map, validate, emit, checkpoint.
        
        Returns:
            Number of events processed
        """
        logger.info(f"Starting connector: {self.source_id}")
        
        # Load checkpoint (best-effort; proceeds even if none exists)
        last_state = self.load_checkpoint()
        if last_state:
            logger.info(f"Resuming from checkpoint: {last_state}")
        
        processed = 0
        batch = []
        batch_size = self.config.get("batch_size", 1000)
        
        last_event_time_ms: Optional[int] = None
        try:
            for raw_event in self.pull_or_subscribe():
                # Map to canonical schema
                try:
                    canonical = self.map_to_schema(raw_event)
                except Exception as e:
                    logger.error(f"Mapping error: {e}")
                    continue
                
                # Validate
                if not self.validate_event(canonical):
                    continue
                
                last_event_time_ms = canonical.get("event_time_utc")
                batch.append(canonical)
                
                # Emit batch when threshold reached
                if len(batch) >= batch_size:
                    count = self.emit(iter(batch))
                    processed += count
                    
                    # Checkpoint progress
                    checkpoint_payload = {
                        "last_event_time": canonical["event_time_utc"],
                        "status": "success",
                        "metadata": {
                            "batch_size": len(batch),
                            "processed_total": processed
                        }
                    }
                    self.checkpoint(checkpoint_payload)
                    
                    batch = []
            
            # Emit remaining records (final flush)
            if batch:
                count = self.emit(iter(batch))
                processed += count

                if last_event_time_ms is not None:
                    checkpoint_payload = {
                        "last_event_time": last_event_time_ms,
                        "status": "success",
                        "metadata": {
                            "batch_size": len(batch),
                            "processed_total": processed
                        }
                    }
                    self.checkpoint(checkpoint_payload)
        
        except Exception as e:
            logger.error(f"Connector error: {e}")
            error_state = {
                "status": "error",
                "last_event_time": last_event_time_ms or self.checkpoint_state.get("last_event_time"),
                "metadata": {
                    "error_message": str(e)
                }
            }
            try:
                self.checkpoint(error_state)
            except Exception:
                logger.exception("Failed to persist error checkpoint state")
            raise
        finally:
            self._loop.run_until_complete(self._close_db())
        
        logger.info(f"Connector finished: {processed} events processed")
        return processed

    async def _get_pool(self) -> asyncpg.pool.Pool:
        if self._db_pool is None:
            self._db_pool = await asyncpg.create_pool(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                command_timeout=60,
                max_size=self.db_config.get("pool_max_size", 10),
                min_size=self.db_config.get("pool_min_size", 1),
                statement_cache_size=self.db_config.get("statement_cache_size", 0),
                server_settings={
                    "application_name": self.db_app_name
                }
            )
        return self._db_pool

    async def _checkpoint_async(self, state: Dict[str, Any]) -> None:
        pool = await self._get_pool()
        status = state.get("status", "success")
        metadata = state.get("metadata", {})
        error_count = state.get("error_count")

        last_event_time_value = state.get("last_event_time")
        last_event_time_dt: Optional[datetime] = None

        if last_event_time_value is not None:
            if isinstance(last_event_time_value, (int, float)):
                last_event_time_dt = datetime.fromtimestamp(last_event_time_value / 1000, tz=timezone.utc)
            elif isinstance(last_event_time_value, str):
                try:
                    last_event_time_dt = datetime.fromisoformat(last_event_time_value)
                except ValueError:
                    last_event_time_dt = None
            elif isinstance(last_event_time_value, datetime):
                last_event_time_dt = last_event_time_value.astimezone(timezone.utc)

        state_payload = state.copy()

        async with pool.acquire() as conn:
            async with conn.transaction():
                prev = await conn.fetchrow(
                    "SELECT error_count FROM connector_checkpoints WHERE connector_id = $1",
                    self.source_id,
                )

                if error_count is None:
                    previous_errors = prev["error_count"] if prev else 0
                    if status == "error":
                        error_count = previous_errors + 1
                    else:
                        error_count = 0

                await conn.execute(
                    """
                    INSERT INTO connector_checkpoints (
                        connector_id,
                        last_event_time,
                        last_successful_run,
                        state,
                        error_count,
                        metadata,
                        updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
                    ON CONFLICT (connector_id)
                    DO UPDATE SET
                        last_event_time = EXCLUDED.last_event_time,
                        last_successful_run = EXCLUDED.last_successful_run,
                        state = EXCLUDED.state,
                        error_count = EXCLUDED.error_count,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    self.source_id,
                    last_event_time_dt,
                    datetime.now(timezone.utc) if status == "success" else None,
                    json.dumps(state_payload),
                    error_count,
                    json.dumps(metadata),
                )

                await conn.execute(
                    """
                    INSERT INTO connector_checkpoint_history (
                        connector_id,
                        state,
                        metadata,
                        status
                    ) VALUES ($1, $2, $3, $4)
                    """,
                    self.source_id,
                    json.dumps(state_payload),
                    json.dumps(metadata),
                    status,
                )

    async def _load_checkpoint_async(self) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    last_event_time,
                    last_successful_run,
                    state,
                    error_count,
                    metadata
                FROM connector_checkpoints
                WHERE connector_id = $1
                """,
                self.source_id,
            )

            if row is None:
                return None

            state_payload = row["state"] or {}
            metadata = row["metadata"] or {}

            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            if isinstance(state_payload, str):
                state_payload = json.loads(state_payload)

            last_event_dt = row["last_event_time"]
            if last_event_dt and state_payload.get("last_event_time") is None:
                state_payload["last_event_time"] = int(last_event_dt.timestamp() * 1000)

            state_payload["last_successful_run"] = (
                row["last_successful_run"].isoformat()
                if row["last_successful_run"]
                else None
            )
            state_payload["error_count"] = row["error_count"]
            state_payload["metadata"] = metadata

            return state_payload

    async def _close_db(self) -> None:
        if self._db_pool is not None:
            await self._db_pool.close()
            self._db_pool = None
