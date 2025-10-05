"""
Rate limiting middleware for API Gateway.

Uses slowapi for flexible rate limiting per endpoint and user.
"""
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

logger = logging.getLogger(__name__)

# Create limiter with remote address as key
limiter = Limiter(key_func=get_remote_address)

# Rate limit tiers
RATE_LIMITS = {
    "public": "100/minute",           # Public endpoints
    "authenticated": "1000/minute",   # Authenticated users
    "heavy": "10/minute",             # Heavy queries (large data)
    "cache_write": "5/minute",        # Cache warming operations
    "stream": "10/minute",            # Stream subscriptions
}


def get_rate_limit(tier: str = "authenticated") -> str:
    """
    Get rate limit string for tier.
    
    Args:
        tier: Rate limit tier name
    
    Returns:
        str: Rate limit string (e.g., "1000/minute")
    """
    return RATE_LIMITS.get(tier, RATE_LIMITS["authenticated"])


def add_rate_limiting(app):
    """
    Add rate limiting to FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    
    logger.info("Rate limiting enabled")

