"""
HTTP client for Metrics Service.
"""
import logging
import os
import time
from functools import wraps

import httpx

logger = logging.getLogger(__name__)

METRICS_SERVICE_URL = os.getenv("METRICS_SERVICE_URL", "http://metrics-service:8012")
SERVICE_NAME = "api-gateway"


async def track_request(
    endpoint: str,
    method: str = "GET",
    status: int = 200,
):
    """
    Track API request (fire-and-forget).
    
    Args:
        endpoint: Endpoint path
        method: HTTP method
        status: HTTP status code
    """
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.post(
                f"{METRICS_SERVICE_URL}/metrics/track",
                json={
                    "endpoint": endpoint,
                    "method": method,
                    "status": status,
                    "service": SERVICE_NAME
                }
            )
    except Exception as e:
        # Don't fail requests if metrics fail
        logger.debug(f"Metrics tracking failed: {e}")


async def track_latency(endpoint: str, duration_seconds: float):
    """
    Track request latency (fire-and-forget).
    
    Args:
        endpoint: Endpoint path
        duration_seconds: Request duration in seconds
    """
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.post(
                f"{METRICS_SERVICE_URL}/metrics/latency",
                json={
                    "endpoint": endpoint,
                    "duration_seconds": duration_seconds,
                    "service": SERVICE_NAME
                }
            )
    except Exception as e:
        logger.debug(f"Latency tracking failed: {e}")


def track_endpoint_latency(endpoint: str):
    """
    Decorator to track endpoint latency.
    
    Usage:
        @track_endpoint_latency("/api/v1/instruments")
        async def get_instruments():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                await track_latency(endpoint, duration)
        return wrapper
    return decorator

