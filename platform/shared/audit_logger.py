"""
Centralized Audit Logging Library for 254Carbon Platform

Overview
--------
Provides structured JSON logging and basic anomaly detection utilities with
distributed tracing context. Intended to support SOC 2 controls around audit
trails, security monitoring, and incident investigation.

Notes
-----
- Storage: includes a database logger that writes to PostgreSQL (via asyncpg).
- Context: uses contextvars to propagate request/trace IDs across coroutines.
- Safety: audit logging must never break business logic; failures are caught
  and logged but not re‑raised.
"""
import logging
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from contextvars import ContextVar
import asyncpg

logger = logging.getLogger(__name__)

# Context variables for distributed tracing propagated implicitly in async tasks
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='')
trace_id_ctx: ContextVar[str] = ContextVar('trace_id', default='')


class StructuredLogger:
    """Structured JSON logging for audit and security events.

    Emits JSON lines that can be shipped to centralized logging backends
    (e.g., Loki, ELK) and correlated via request/trace IDs.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
    
    def _log_structured(
        self,
        level: str,
        event_type: str,
        message: str,
        **kwargs
    ):
        """Log a structured JSON event to the configured logger."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": self.service_name,
            "level": level,
            "event_type": event_type,
            "message": message,
            "request_id": request_id_ctx.get(),
            "trace_id": trace_id_ctx.get(),
            **kwargs
        }
        
        # Remove None values
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_entry))
    
    def audit(
        self,
        action: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        success: bool,
        **details
    ):
        """Log an audit event describing a user action on a resource."""
        self._log_structured(
            level="INFO",
            event_type="audit",
            message=f"{action} {resource_type}:{resource_id}",
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            **details
        )
    
    def security(
        self,
        event: str,
        severity: str,
        user_id: Optional[str] = None,
        **details
    ):
        """Log a security event with severity label (low/medium/high/critical)."""
        self._log_structured(
            level="WARNING" if severity in ["medium", "high", "critical"] else "INFO",
            event_type="security",
            message=event,
            severity=severity,
            user_id=user_id,
            **details
        )
    
    def performance(
        self,
        operation: str,
        duration_ms: float,
        **details
    ):
        """Log a performance metric for the given operation."""
        self._log_structured(
            level="INFO",
            event_type="performance",
            message=f"{operation} completed in {duration_ms:.2f}ms",
            operation=operation,
            duration_ms=duration_ms,
            **details
        )
    
    def error(
        self,
        error_type: str,
        message: str,
        **details
    ):
        """Log an error event without raising exceptions."""
        self._log_structured(
            level="ERROR",
            event_type="error",
            message=message,
            error_type=error_type,
            **details
        )


class DatabaseAuditLogger:
    """Database-backed audit logger for compliance and investigation.

    Persists events to PostgreSQL (schema/table assumed as ``pg.audit_log``)
    and mirrors them to structured logs for aggregation. The database writes are
    best‑effort; failures are recorded but do not propagate to callers.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool
        self.structured_logger = StructuredLogger("audit_db")
    
    async def log_event(
        self,
        user_id: str,
        tenant_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: str,
        ip_address: Optional[str],
        user_agent: Optional[str],
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a single audit event to the database and structured logs.

        Notes
        -----
        - The SQL uses parameter binding to avoid injection risks.
        - Consider partitioning/TTL strategies for large audit tables.
        """
        try:
            async with self.pool.acquire() as conn:
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
                    ip_address,
                    user_agent,
                    success,
                    json.dumps(details) if details else None,
                )
            
            # Also log structured event for immediate visibility in logs
            self.structured_logger.audit(
                action=action,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                success=success,
                tenant_id=tenant_id,
                ip_address=ip_address,
            )
            
        except Exception as e:
            # Never fail caller operations due to audit logging
            logger.error(f"Failed to log audit event: {e}")
            self.structured_logger.error(
                error_type="audit_logging_failure",
                message=str(e),
                user_id=user_id,
                action=action,
            )
    
    async def log_authentication(
        self,
        user_id: str,
        tenant_id: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
        success: bool,
        failure_reason: Optional[str] = None,
    ):
        """Log authentication attempt and flag failures as security events."""
        details = {
            "failure_reason": failure_reason,
            "request_id": request_id_ctx.get(),
        }
        
        await self.log_event(
            user_id=user_id,
            tenant_id=tenant_id,
            action="authenticate",
            resource_type="auth",
            resource_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details if failure_reason else None,
        )
        
        # Log security event for failed authentication
        if not success:
            self.structured_logger.security(
                event="authentication_failure",
                severity="medium",
                user_id=user_id,
                ip_address=ip_address,
                reason=failure_reason,
            )
    
    async def log_authorization_failure(
        self,
        user_id: str,
        tenant_id: Optional[str],
        resource_type: str,
        resource_id: str,
        ip_address: Optional[str],
        reason: str,
    ):
        """Log authorization failure (potential security event)."""
        await self.log_event(
            user_id=user_id,
            tenant_id=tenant_id,
            action="authorize_denied",
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=None,
            success=False,
            details={"reason": reason},
        )
        
        self.structured_logger.security(
            event="authorization_denied",
            severity="medium",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            reason=reason,
        )
    
    async def log_sensitive_data_access(
        self,
        user_id: str,
        tenant_id: Optional[str],
        resource_type: str,
        resource_id: str,
        ip_address: Optional[str],
        records_accessed: int,
    ):
        """Log access to sensitive data (PII, pricing, etc.)."""
        await self.log_event(
            user_id=user_id,
            tenant_id=tenant_id,
            action="read_sensitive",
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=None,
            success=True,
            details={
                "records_accessed": records_accessed,
                "data_classification": "sensitive",
            },
        )
    
    async def log_configuration_change(
        self,
        user_id: str,
        tenant_id: Optional[str],
        config_key: str,
        old_value: Any,
        new_value: Any,
        ip_address: Optional[str],
    ):
        """Log configuration changes for audit trail."""
        await self.log_event(
            user_id=user_id,
            tenant_id=tenant_id,
            action="config_change",
            resource_type="configuration",
            resource_id=config_key,
            ip_address=ip_address,
            user_agent=None,
            success=True,
            details={
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
        )
        
        self.structured_logger.security(
            event="configuration_changed",
            severity="low",
            user_id=user_id,
            config_key=config_key,
        )


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a unique trace ID for distributed tracing."""
    return str(uuid.uuid4())


def set_request_context(request_id: str, trace_id: Optional[str] = None):
    """Set request context for logging (idempotent for missing trace IDs)."""
    request_id_ctx.set(request_id)
    if trace_id:
        trace_id_ctx.set(trace_id)
    else:
        trace_id_ctx.set(generate_trace_id())


def get_request_context() -> Dict[str, str]:
    """Get the current request/trace context as a dict."""
    return {
        "request_id": request_id_ctx.get(),
        "trace_id": trace_id_ctx.get(),
    }


# Security event patterns for monitoring
SECURITY_PATTERNS = {
    "brute_force": "Multiple failed authentication attempts",
    "privilege_escalation": "Attempt to access unauthorized resources",
    "data_exfiltration": "Unusual volume of data export",
    "suspicious_ip": "Access from suspicious IP address",
    "after_hours": "Access outside normal business hours",
}


class SecurityEventDetector:
    """Detect suspicious patterns in audit logs using simple heuristics.

    This is not a full SIEM; it provides lightweight checks that can be
    scheduled periodically or invoked on specific triggers.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool
        self.logger = StructuredLogger("security_detector")
    
    async def check_brute_force(self, user_id: str, window_minutes: int = 15) -> bool:
        """Check for brute force authentication attempts over a time window."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM pg.audit_log
                WHERE user_id = $1
                  AND action = 'authenticate'
                  AND success = false
                  AND timestamp > NOW() - INTERVAL '$2 minutes'
                """,
                user_id,
                window_minutes,
            )
            
            if result and result >= 5:
                self.logger.security(
                    event="brute_force_detected",
                    severity="high",
                    user_id=user_id,
                    failed_attempts=result,
                    window_minutes=window_minutes,
                )
                return True
        
        return False
    
    async def check_unusual_data_export(
        self,
        user_id: str,
        current_export_size: int,
    ) -> bool:
        """Check for unusually large data exports."""
        async with self.pool.acquire() as conn:
            # Get average export size for user
            result = await conn.fetchrow(
                """
                SELECT AVG((details->>'file_size_bytes')::bigint) as avg_size,
                       STDDEV((details->>'file_size_bytes')::bigint) as std_size
                FROM pg.audit_log
                WHERE user_id = $1
                  AND action = 'export'
                  AND success = true
                  AND timestamp > NOW() - INTERVAL '30 days'
                """,
                user_id,
            )
            
            if result and result['avg_size']:
                avg_size = result['avg_size']
                std_size = result['std_size'] or 0
                
                # Flag if export is > 3 standard deviations above average
                if current_export_size > (avg_size + 3 * std_size):
                    self.logger.security(
                        event="unusual_data_export",
                        severity="medium",
                        user_id=user_id,
                        current_size=current_export_size,
                        average_size=avg_size,
                    )
                    return True
        
        return False


