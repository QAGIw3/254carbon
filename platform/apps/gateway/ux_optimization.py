"""
UX optimization middleware and utilities for improved loading states and error handling.
"""
import logging
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class UXOptimizationMiddleware(BaseHTTPMiddleware):
    """Middleware for UX optimizations including loading states and error handling."""

    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.request_timings: Dict[str, float] = {}

    async def dispatch(self, request: Request, call_next):
        """Process request with UX optimizations."""
        start_time = time.time()
        request_id = f"{request.client.host}:{request.url.path}:{start_time}"

        try:
            # Add request timing header for frontend
            response = await call_next(request)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Add timing headers for frontend loading states
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time".3f"}s"

            # Add loading state hints based on processing time
            if processing_time > self.slow_request_threshold:
                response.headers["X-Loading-State"] = "slow"
                logger.warning(f"Slow request detected: {request.url.path} took {processing_time".3f"}s")
            else:
                response.headers["X-Loading-State"] = "fast"

            # Add cache status if applicable
            if hasattr(request.state, 'cache_status'):
                response.headers["X-Cache-Status"] = request.state.cache_status

            return response

        except HTTPException as e:
            # Enhanced error response with UX-friendly formatting
            processing_time = time.time() - start_time

            error_response = {
                "error": {
                    "code": e.status_code,
                    "message": e.detail,
                    "request_id": request_id,
                    "processing_time": f"{processing_time".3f"}s",
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path),
                    "method": request.method
                }
            }

            # Add retry information for certain error types
            if e.status_code in [500, 502, 503, 504]:
                error_response["error"]["retry_after"] = "30"  # Suggest retry after 30 seconds

            return JSONResponse(
                status_code=e.status_code,
                content=error_response,
                headers={
                    "X-Request-ID": request_id,
                    "X-Error-Type": "server_error" if e.status_code >= 500 else "client_error"
                }
            )

        except Exception as e:
            # Catch-all for unexpected errors
            processing_time = time.time() - start_time

            logger.error(f"Unexpected error in {request.url.path}: {e}")

            error_response = {
                "error": {
                    "code": 500,
                    "message": "An unexpected error occurred. Please try again.",
                    "request_id": request_id,
                    "processing_time": f"{processing_time".3f"}s",
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path),
                    "method": request.method,
                    "retry_after": "30"
                }
            }

            return JSONResponse(
                status_code=500,
                content=error_response,
                headers={
                    "X-Request-ID": request_id,
                    "X-Error-Type": "unexpected_error"
                }
            )


class LoadingStateManager:
    """Manage loading states for different types of requests."""

    def __init__(self):
        self.loading_states = {
            "instruments": {"estimated_time": 0.1, "cache_ttl": 900},  # 15 minutes
            "price_ticks": {"estimated_time": 0.5, "cache_ttl": 60},   # 1 minute
            "forward_curves": {"estimated_time": 2.0, "cache_ttl": 300}, # 5 minutes
            "fundamentals": {"estimated_time": 1.5, "cache_ttl": 1800},  # 30 minutes
            "reports": {"estimated_time": 5.0, "cache_ttl": 0},        # No cache
            "exports": {"estimated_time": 10.0, "cache_ttl": 0},       # No cache
        }

    def get_loading_config(self, endpoint: str) -> Dict[str, Any]:
        """Get loading configuration for an endpoint."""
        return self.loading_states.get(endpoint, {
            "estimated_time": 1.0,
            "cache_ttl": 300
        })

    def is_cacheable(self, endpoint: str) -> bool:
        """Check if endpoint response should be cached."""
        config = self.get_loading_config(endpoint)
        return config["cache_ttl"] > 0

    def get_estimated_time(self, endpoint: str) -> float:
        """Get estimated processing time for loading states."""
        config = self.get_loading_config(endpoint)
        return config["estimated_time"]


class ErrorHandler:
    """Enhanced error handling with UX-friendly responses."""

    @staticmethod
    def format_validation_error(error_detail: str) -> Dict[str, Any]:
        """Format validation errors for better UX."""
        return {
            "type": "validation_error",
            "message": "The request data is invalid. Please check your input.",
            "details": error_detail,
            "suggestions": [
                "Check that all required fields are provided",
                "Ensure date formats are YYYY-MM-DD",
                "Verify instrument IDs are valid",
                "Check that numeric values are within acceptable ranges"
            ]
        }

    @staticmethod
    def format_timeout_error(timeout_seconds: int) -> Dict[str, Any]:
        """Format timeout errors for better UX."""
        return {
            "type": "timeout_error",
            "message": f"Request timed out after {timeout_seconds} seconds.",
            "details": "The server is processing a large dataset or experiencing high load.",
            "suggestions": [
                "Try reducing the date range or number of instruments",
                "Use cached data if available",
                "Schedule large exports for off-peak hours",
                "Contact support if the issue persists"
            ],
            "retry_after": str(timeout_seconds * 2)
        }

    @staticmethod
    def format_rate_limit_error(limit_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format rate limit errors for better UX."""
        return {
            "type": "rate_limit_error",
            "message": "Too many requests. Please slow down.",
            "details": f"Rate limit: {limit_info.get('requests_per_minute', 'unknown')} requests per minute",
            "suggestions": [
                "Wait a few minutes before retrying",
                "Use cached data when possible",
                "Consider batching multiple requests",
                "Upgrade to a higher tier plan for increased limits"
            ],
            "retry_after": "300"  # 5 minutes
        }


# Utility functions for response optimization

async def add_loading_metadata(response_data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    """Add loading state metadata to API responses."""
    loading_manager = LoadingStateManager()

    response_data["_metadata"] = {
        "loading_config": loading_manager.get_loading_config(endpoint),
        "estimated_time": loading_manager.get_estimated_time(endpoint),
        "cacheable": loading_manager.is_cacheable(endpoint),
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint
    }

    return response_data


async def create_progressive_response(
    request: Request,
    total_items: int,
    items_per_chunk: int = 100
) -> Dict[str, Any]:
    """Create response structure for progressive loading."""
    return {
        "total_items": total_items,
        "chunk_size": items_per_chunk,
        "chunks_available": (total_items + items_per_chunk - 1) // items_per_chunk,
        "supports_progressive": True,
        "estimated_load_time": f"{(total_items / items_per_chunk) * 0.5".1f"} seconds",
        "suggestions": [
            "Use pagination for large datasets",
            "Consider filtering to reduce data size",
            "Cache frequently accessed data"
        ]
    }


# Background task for monitoring slow requests
async def monitor_slow_requests():
    """Background task to monitor and log slow requests."""
    while True:
        try:
            # This would integrate with your metrics collection system
            # For now, just log slow request patterns
            await asyncio.sleep(300)  # Check every 5 minutes

        except Exception as e:
            logger.error(f"Error in slow request monitoring: {e}")
            await asyncio.sleep(60)


# Integration helper for FastAPI
def add_ux_middleware(app):
    """Add UX optimization middleware to FastAPI app."""
    app.add_middleware(UXOptimizationMiddleware, slow_request_threshold=1.0)

    # Start background monitoring
    asyncio.create_task(monitor_slow_requests())

    logger.info("UX optimization middleware added to application")

