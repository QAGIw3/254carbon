"""
Base connector SDK for data ingestion.

All data source connectors implement the Ingestor interface.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor

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

        # PostgreSQL connection for checkpoints
        self.db_config = config.get("database", {})
        self.db_host = self.db_config.get("host", "postgresql")
        self.db_port = self.db_config.get("port", 5432)
        self.db_name = self.db_config.get("database", "market_intelligence")
        self.db_user = self.db_config.get("user", "postgres")
        self.db_password = self.db_config.get("password", "postgres")

        # Initialize database connection
        self._init_db()
    
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
        Save checkpoint state for resume/recovery.

        Args:
            state: State dict to persist (e.g., last_timestamp, last_offset)
        """
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            conn.autocommit = True

            # Convert state to JSON for storage
            state_json = json.dumps(state)

            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO connector_checkpoints (source_id, checkpoint_data, last_updated)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (source_id)
                    DO UPDATE SET checkpoint_data = %s, last_updated = CURRENT_TIMESTAMP
                """, (self.source_id, state_json, state_json))

            conn.close()
            logger.debug(f"Checkpoint saved for {self.source_id}: {state}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            # Don't raise exception - checkpoint failures shouldn't stop ingestion
    
    def validate_event(self, event: Dict[str, Any]) -> bool:
        """
        Validate event against data quality rules.

        Args:
            event: Canonical schema event

        Returns:
            True if valid, False otherwise
        """
        # Basic validation
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

    def _init_db(self) -> None:
        """Initialize database connection and create checkpoint table if needed."""
        try:
            # Connect to database
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            conn.autocommit = True

            # Create checkpoint table if it doesn't exist
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS connector_checkpoints (
                        source_id VARCHAR(100) PRIMARY KEY,
                        checkpoint_data JSONB NOT NULL,
                        last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            conn.close()
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
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT checkpoint_data FROM connector_checkpoints WHERE source_id = %s",
                    (self.source_id,)
                )
                row = cursor.fetchone()

            conn.close()

            if row:
                self.checkpoint_state = row['checkpoint_data']
                logger.info(f"Loaded checkpoint for {self.source_id}: {self.checkpoint_state}")
                return self.checkpoint_state
            else:
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
        
        # Load checkpoint
        last_state = self.load_checkpoint()
        if last_state:
            logger.info(f"Resuming from checkpoint: {last_state}")
        
        processed = 0
        batch = []
        batch_size = self.config.get("batch_size", 1000)
        
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
                
                batch.append(canonical)
                
                # Emit batch
                if len(batch) >= batch_size:
                    count = self.emit(iter(batch))
                    processed += count
                    
                    # Checkpoint
                    self.checkpoint({"last_event_time": canonical["event_time_utc"]})
                    
                    batch = []
            
            # Emit remaining
            if batch:
                count = self.emit(iter(batch))
                processed += count
        
        except Exception as e:
            logger.error(f"Connector error: {e}")
            raise
        
        logger.info(f"Connector finished: {processed} events processed")
        return processed

