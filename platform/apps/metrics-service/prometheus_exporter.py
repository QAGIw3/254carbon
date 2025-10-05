"""
Prometheus metrics collection and export.
"""
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

logger = logging.getLogger(__name__)

# Request counters by service
requests_total = Counter(
    "service_requests_total",
    "Total requests across all services",
    ["service", "endpoint", "method", "status"],
)

# Latency histogram by service
request_latency = Histogram(
    "service_request_duration_seconds",
    "Request latency across all services",
    ["service", "endpoint"],
)

# Active connections by service
active_connections = Gauge(
    "service_active_connections",
    "Number of active connections by type",
    ["service", "type"],
)

# Stream-specific metrics
stream_latency = Histogram(
    "stream_latency_seconds",
    "Stream latency from source to client",
    ["service", "source"],
)


def track_request_metric(
    endpoint: str,
    method: str = "GET",
    status: str = "200",
    service: str = "unknown",
):
    """
    Track an API request.
    
    Args:
        endpoint: Endpoint path
        method: HTTP method
        status: HTTP status code
        service: Service name
    """
    try:
        requests_total.labels(
            service=service,
            endpoint=endpoint,
            method=method,
            status=status,
        ).inc()
    except Exception as e:
        logger.error(f"Error tracking request metric: {e}")


def track_latency_metric(
    endpoint: str,
    duration: float,
    service: str = "unknown",
):
    """
    Track request latency.
    
    Args:
        endpoint: Endpoint path
        duration: Request duration in seconds
        service: Service name
    """
    try:
        request_latency.labels(
            service=service,
            endpoint=endpoint,
        ).observe(duration)
    except Exception as e:
        logger.error(f"Error tracking latency metric: {e}")


def track_connection_metric(
    connection_type: str,
    delta: int,
    service: str = "unknown",
):
    """
    Track active connections.
    
    Args:
        connection_type: Type of connection (websocket, http, etc.)
        delta: Change in connections (+1 for new, -1 for closed)
        service: Service name
    """
    try:
        if delta > 0:
            active_connections.labels(
                service=service,
                type=connection_type,
            ).inc(delta)
        elif delta < 0:
            active_connections.labels(
                service=service,
                type=connection_type,
            ).dec(abs(delta))
    except Exception as e:
        logger.error(f"Error tracking connection metric: {e}")


def get_metrics_text() -> str:
    """
    Get metrics in Prometheus text format.
    
    Returns:
        str: Prometheus-formatted metrics
    """
    try:
        return generate_latest(REGISTRY).decode("utf-8")
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return ""


def clear_all_metrics():
    """
    Clear all metrics (for testing).
    
    WARNING: This is destructive and should only be used in testing.
    """
    # Note: Prometheus client doesn't provide a built-in way to clear metrics
    # This is a placeholder for testing scenarios
    logger.warning("Metrics clearing requested (not fully implemented)")

