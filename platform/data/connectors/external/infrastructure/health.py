"""
Infrastructure Connector Health Check Service
---------------------------------------------

Provides health check endpoints for infrastructure connectors
with detailed status information and diagnostics.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import FastAPI, Response
from pydantic import BaseModel
import asyncpg
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    name: str
    status: HealthStatus
    message: Optional[str] = None
    last_check: datetime
    metadata: Dict[str, Any] = {}


class ConnectorHealth(BaseModel):
    connector_id: str
    status: HealthStatus
    uptime_seconds: float
    last_successful_run: Optional[datetime]
    last_error: Optional[str]
    records_processed_total: int
    data_freshness_hours: Optional[float]
    components: List[ComponentHealth]
    version: str = "1.0.0"


class HealthCheckService:
    """Service for monitoring connector health."""
    
    def __init__(self, connector_id: str, db_config: Dict[str, Any]):
        self.connector_id = connector_id
        self.db_config = db_config
        self.start_time = datetime.now(timezone.utc)
        self._pg_pool = None
        self._last_health_check = datetime.now(timezone.utc)
        self._health_cache: Optional[ConnectorHealth] = None
        self._cache_ttl = 30  # seconds
    
    async def get_pg_pool(self):
        """Get or create PostgreSQL connection pool."""
        if self._pg_pool is None:
            self._pg_pool = await asyncpg.create_pool(
                host=self.db_config.get("host", "postgresql"),
                port=self.db_config.get("port", 5432),
                database=self.db_config.get("database", "market_intelligence"),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", "postgres"),
                min_size=1,
                max_size=5,
            )
        return self._pg_pool
    
    async def check_health(self) -> ConnectorHealth:
        """Perform comprehensive health check."""
        
        # Check cache
        if self._health_cache and (datetime.now(timezone.utc) - self._last_health_check).seconds < self._cache_ttl:
            return self._health_cache
        
        components = []
        overall_status = HealthStatus.HEALTHY
        
        # Check database connectivity
        db_health = await self._check_database()
        components.append(db_health)
        if db_health.status != HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED
        
        # Check checkpoint status
        checkpoint_health = await self._check_checkpoint()
        components.append(checkpoint_health)
        if checkpoint_health.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif checkpoint_health.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED
        
        # Check data freshness
        freshness_health = await self._check_data_freshness()
        components.append(freshness_health)
        if freshness_health.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif freshness_health.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED
        
        # Get connector metrics
        metrics = await self._get_connector_metrics()
        
        # Calculate uptime
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        health = ConnectorHealth(
            connector_id=self.connector_id,
            status=overall_status,
            uptime_seconds=uptime,
            last_successful_run=metrics.get("last_successful_run"),
            last_error=metrics.get("last_error"),
            records_processed_total=metrics.get("records_processed", 0),
            data_freshness_hours=metrics.get("data_freshness_hours"),
            components=components,
        )
        
        # Update cache
        self._health_cache = health
        self._last_health_check = datetime.now(timezone.utc)
        
        return health
    
    async def _check_database(self) -> ComponentHealth:
        """Check database connectivity."""
        try:
            pool = await self.get_pg_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    return ComponentHealth(
                        name="database",
                        status=HealthStatus.HEALTHY,
                        message="Database connection successful",
                        last_check=datetime.now(timezone.utc),
                    )
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                last_check=datetime.now(timezone.utc),
            )
        
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message="Unknown database error",
            last_check=datetime.now(timezone.utc),
        )
    
    async def _check_checkpoint(self) -> ComponentHealth:
        """Check checkpoint status."""
        try:
            pool = await self.get_pg_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT 
                        last_event_time,
                        last_successful_run,
                        error_count,
                        state
                    FROM connector_checkpoints
                    WHERE connector_id = $1
                    """,
                    self.connector_id
                )
                
                if not row:
                    return ComponentHealth(
                        name="checkpoint",
                        status=HealthStatus.DEGRADED,
                        message="No checkpoint found",
                        last_check=datetime.now(timezone.utc),
                    )
                
                error_count = row["error_count"] or 0
                if error_count > 10:
                    return ComponentHealth(
                        name="checkpoint",
                        status=HealthStatus.UNHEALTHY,
                        message=f"High error count: {error_count}",
                        last_check=datetime.now(timezone.utc),
                        metadata={"error_count": error_count},
                    )
                elif error_count > 5:
                    return ComponentHealth(
                        name="checkpoint",
                        status=HealthStatus.DEGRADED,
                        message=f"Elevated error count: {error_count}",
                        last_check=datetime.now(timezone.utc),
                        metadata={"error_count": error_count},
                    )
                
                return ComponentHealth(
                    name="checkpoint",
                    status=HealthStatus.HEALTHY,
                    message="Checkpoint status normal",
                    last_check=datetime.now(timezone.utc),
                    metadata={
                        "error_count": error_count,
                        "last_event_time": row["last_event_time"].isoformat() if row["last_event_time"] else None,
                    },
                )
                
        except Exception as e:
            logger.error(f"Checkpoint health check failed: {e}")
            return ComponentHealth(
                name="checkpoint",
                status=HealthStatus.UNHEALTHY,
                message=f"Checkpoint check failed: {str(e)}",
                last_check=datetime.now(timezone.utc),
            )
    
    async def _check_data_freshness(self) -> ComponentHealth:
        """Check data freshness."""
        try:
            pool = await self.get_pg_pool()
            async with pool.acquire() as conn:
                last_event_time = await conn.fetchval(
                    """
                    SELECT last_event_time
                    FROM connector_checkpoints
                    WHERE connector_id = $1
                    """,
                    self.connector_id
                )
                
                if not last_event_time:
                    return ComponentHealth(
                        name="data_freshness",
                        status=HealthStatus.DEGRADED,
                        message="No data timestamp available",
                        last_check=datetime.now(timezone.utc),
                    )
                
                # Calculate age
                age = datetime.now(timezone.utc) - last_event_time
                hours = age.total_seconds() / 3600
                
                # Define freshness thresholds based on connector type
                freshness_thresholds = {
                    "alsi_lng_inventory": {"warning": 24, "critical": 48},
                    "reexplorer_renewable": {"warning": 168, "critical": 336},  # Weekly updates OK
                    "wri_powerplants": {"warning": 720, "critical": 1440},  # Monthly updates OK
                    "gem_transmission": {"warning": 168, "critical": 336},
                }
                
                thresholds = freshness_thresholds.get(
                    self.connector_id,
                    {"warning": 24, "critical": 72}  # Default
                )
                
                if hours > thresholds["critical"]:
                    return ComponentHealth(
                        name="data_freshness",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Data is {hours:.1f} hours old (critical threshold: {thresholds['critical']}h)",
                        last_check=datetime.now(timezone.utc),
                        metadata={"age_hours": hours},
                    )
                elif hours > thresholds["warning"]:
                    return ComponentHealth(
                        name="data_freshness",
                        status=HealthStatus.DEGRADED,
                        message=f"Data is {hours:.1f} hours old (warning threshold: {thresholds['warning']}h)",
                        last_check=datetime.now(timezone.utc),
                        metadata={"age_hours": hours},
                    )
                
                return ComponentHealth(
                    name="data_freshness",
                    status=HealthStatus.HEALTHY,
                    message=f"Data is {hours:.1f} hours old",
                    last_check=datetime.now(timezone.utc),
                    metadata={"age_hours": hours},
                )
                
        except Exception as e:
            logger.error(f"Data freshness check failed: {e}")
            return ComponentHealth(
                name="data_freshness",
                status=HealthStatus.UNHEALTHY,
                message=f"Freshness check failed: {str(e)}",
                last_check=datetime.now(timezone.utc),
            )
    
    async def _get_connector_metrics(self) -> Dict[str, Any]:
        """Get connector metrics from database."""
        try:
            pool = await self.get_pg_pool()
            async with pool.acquire() as conn:
                # Get checkpoint data
                checkpoint = await conn.fetchrow(
                    """
                    SELECT 
                        last_event_time,
                        last_successful_run,
                        state,
                        metadata
                    FROM connector_checkpoints
                    WHERE connector_id = $1
                    """,
                    self.connector_id
                )
                
                if not checkpoint:
                    return {}
                
                # Parse state for metrics
                state = checkpoint["state"] or {}
                if isinstance(state, str):
                    state = json.loads(state)
                
                metadata = checkpoint["metadata"] or {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                # Calculate data freshness
                data_freshness_hours = None
                if checkpoint["last_event_time"]:
                    age = datetime.now(timezone.utc) - checkpoint["last_event_time"]
                    data_freshness_hours = age.total_seconds() / 3600
                
                # Get record count from history
                record_count = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM connector_checkpoint_history
                    WHERE connector_id = $1 AND status = 'success'
                    """,
                    self.connector_id
                )
                
                return {
                    "last_successful_run": checkpoint["last_successful_run"],
                    "last_error": state.get("error_message"),
                    "records_processed": metadata.get("processed_total", 0),
                    "data_freshness_hours": data_freshness_hours,
                    "checkpoint_count": record_count or 0,
                }
                
        except Exception as e:
            logger.error(f"Failed to get connector metrics: {e}")
            return {}
    
    async def close(self):
        """Close database connections."""
        if self._pg_pool:
            await self._pg_pool.close()


# Create FastAPI app for health endpoints
def create_health_app(connector_id: str, db_config: Dict[str, Any]) -> FastAPI:
    """Create FastAPI app with health check endpoints."""
    
    app = FastAPI(
        title=f"{connector_id} Health Check",
        version="1.0.0",
    )
    
    health_service = HealthCheckService(connector_id, db_config)
    
    @app.get("/health", response_model=ConnectorHealth)
    async def health_check():
        """Get comprehensive health status."""
        return await health_service.check_health()
    
    @app.get("/health/live")
    async def liveness_check():
        """Simple liveness check."""
        return {"status": "alive"}
    
    @app.get("/health/ready")
    async def readiness_check():
        """Readiness check."""
        health = await health_service.check_health()
        if health.status == HealthStatus.UNHEALTHY:
            return Response(
                content=json.dumps({"status": "not ready", "reason": health.status}),
                status_code=503,
            )
        return {"status": "ready"}
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    
    @app.on_event("shutdown")
    async def shutdown():
        """Clean up on shutdown."""
        await health_service.close()
    
    return app
