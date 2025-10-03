"""
Comprehensive audit logging for all data access and mutations.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import Request

from .db import get_postgres_pool

logger = logging.getLogger(__name__)


class AuditLogger:
    """Centralized audit logging service."""
    
    @staticmethod
    async def log_access(
        user_id: str,
        tenant_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: str,
        request: Request,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log data access or mutation event.
        
        Args:
            user_id: User identifier
            tenant_id: Tenant/organization ID
            action: Action performed (read, write, delete, export, etc.)
            resource_type: Type of resource (instrument, curve, scenario, etc.)
            resource_id: Specific resource identifier
            request: FastAPI request object
            success: Whether action succeeded
            details: Additional context (query params, filters, etc.)
        """
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO pg.audit_log
                    (timestamp, user_id, tenant_id, action, resource_type, 
                     resource_id, ip_address, user_agent, success, details)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    datetime.utcnow(),
                    user_id,
                    tenant_id,
                    action,
                    resource_type,
                    resource_id,
                    request.client.host if request.client else None,
                    request.headers.get("user-agent"),
                    success,
                    details,
                )
        except Exception as e:
            # Never fail request due to audit logging issues
            logger.error(f"Audit logging failed: {e}")
    
    @staticmethod
    async def log_authentication(
        user_id: str,
        tenant_id: Optional[str],
        request: Request,
        success: bool,
        reason: Optional[str] = None,
    ):
        """Log authentication attempt."""
        await AuditLogger.log_access(
            user_id=user_id,
            tenant_id=tenant_id,
            action="authenticate",
            resource_type="auth",
            resource_id=user_id,
            request=request,
            success=success,
            details={"reason": reason} if reason else None,
        )
    
    @staticmethod
    async def log_api_call(
        user_id: str,
        tenant_id: Optional[str],
        endpoint: str,
        request: Request,
        success: bool,
        response_code: int,
        latency_ms: float,
        records_returned: Optional[int] = None,
    ):
        """Log API endpoint access."""
        await AuditLogger.log_access(
            user_id=user_id,
            tenant_id=tenant_id,
            action="api_call",
            resource_type="endpoint",
            resource_id=endpoint,
            request=request,
            success=success,
            details={
                "response_code": response_code,
                "latency_ms": latency_ms,
                "records_returned": records_returned,
                "query_params": dict(request.query_params),
            },
        )
    
    @staticmethod
    async def log_data_export(
        user_id: str,
        tenant_id: Optional[str],
        instrument_ids: list,
        format: str,
        request: Request,
        success: bool,
        file_size_bytes: Optional[int] = None,
    ):
        """Log data export operation."""
        await AuditLogger.log_access(
            user_id=user_id,
            tenant_id=tenant_id,
            action="export",
            resource_type="data",
            resource_id=",".join(instrument_ids[:5]),  # First 5 IDs
            request=request,
            success=success,
            details={
                "instrument_count": len(instrument_ids),
                "format": format,
                "file_size_bytes": file_size_bytes,
            },
        )
    
    @staticmethod
    async def log_scenario_run(
        user_id: str,
        tenant_id: Optional[str],
        scenario_id: str,
        run_id: str,
        request: Request,
        success: bool,
    ):
        """Log scenario execution."""
        await AuditLogger.log_access(
            user_id=user_id,
            tenant_id=tenant_id,
            action="run_scenario",
            resource_type="scenario",
            resource_id=scenario_id,
            request=request,
            success=success,
            details={"run_id": run_id},
        )


async def audit_middleware(request: Request, call_next):
    """Middleware to automatically log all API requests."""
    import time
    
    start_time = time.time()
    
    # Extract user from request state (set by auth middleware)
    user = getattr(request.state, "user", None)
    
    try:
        response = await call_next(request)
        success = 200 <= response.status_code < 400
        latency_ms = (time.time() - start_time) * 1000
        
        if user and request.url.path.startswith("/api/"):
            await AuditLogger.log_api_call(
                user_id=user.get("sub", "unknown"),
                tenant_id=user.get("tenant_id"),
                endpoint=request.url.path,
                request=request,
                success=success,
                response_code=response.status_code,
                latency_ms=latency_ms,
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise

