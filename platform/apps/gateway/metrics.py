"""
Prometheus metrics for API Gateway.

Exposes counters/histograms/gauges and lightweight helpers to increment
and time requests. The helpers are intentionally minimal so they can be
swapped or augmented by middleware in larger deployments.
"""
import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

# Request counters
requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

# Latency histogram
request_latency = Histogram(
    "api_request_duration_seconds",
    "API request latency",
    ["endpoint"],
)

# Active connections
active_connections = Gauge(
    "api_active_connections",
    "Number of active connections",
    ["type"],
)

# Stream latency
stream_latency = Histogram(
    "stream_latency_seconds",
    "Stream latency from vendor tick to client",
    ["source"],
)


def track_request(endpoint: str):
    """Track API request."""
    requests_total.labels(endpoint=endpoint, method="GET", status="200").inc()


def track_latency(endpoint: str):
    """Decorator to track request latency."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                request_latency.labels(endpoint=endpoint).observe(duration)
        return wrapper
    return decorator
