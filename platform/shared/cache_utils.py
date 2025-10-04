"""
Intelligent Caching Utilities

Provides Redis-based caching decorators and utilities for
frequently accessed data across services.
"""
import logging
import json
import hashlib
from datetime import timedelta
from typing import Any, Callable, Optional
from functools import wraps

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager with intelligent invalidation."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_ttl: int = 300,  # 5 minutes
        prefix: str = "254carbon",
    ):
        """Initialize cache manager."""
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.default_ttl = default_ttl
        self.prefix = prefix

        # Initialize Redis connection
        try:
            self.client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}/{redis_db}")
        except RedisError as e:
            logger.warning(f"Failed to connect to Redis: {e}. Cache will be disabled.")
            self.client = None

        # Metrics
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def _make_key(self, namespace: str, key: str) -> str:
        """Generate cache key with prefix and namespace."""
        return f"{self.prefix}:{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            return None

        try:
            cache_key = self._make_key(namespace, key)
            value = self.client.get(cache_key)

            if value:
                self.hits += 1
                logger.debug(f"Cache HIT: {cache_key}")
                return json.loads(value)
            else:
                self.misses += 1
                logger.debug(f"Cache MISS: {cache_key}")
                return None

        except RedisError as e:
            self.errors += 1
            logger.warning(f"Redis get error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return None

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache with TTL."""
        if not self.client:
            return False

        try:
            cache_key = self._make_key(namespace, key)
            serialized = json.dumps(value)

            ttl_seconds = ttl or self.default_ttl

            self.client.setex(cache_key, ttl_seconds, serialized)
            logger.debug(f"Cache SET: {cache_key} (TTL={ttl_seconds}s)")
            return True

        except RedisError as e:
            self.errors += 1
            logger.warning(f"Redis set error: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.warning(f"Serialization error: {e}")
            return False

    def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache."""
        if not self.client:
            return False

        try:
            cache_key = self._make_key(namespace, key)
            deleted = self.client.delete(cache_key)
            logger.debug(f"Cache DELETE: {cache_key}")
            return deleted > 0

        except RedisError as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        if not self.client:
            return 0

        try:
            cache_pattern = f"{self.prefix}:{pattern}"
            keys = self.client.keys(cache_pattern)

            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cache INVALIDATE: {deleted} keys matching {cache_pattern}")
                return deleted
            return 0

        except RedisError as e:
            logger.warning(f"Redis pattern delete error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "total_requests": total_requests,
            "hit_rate_pct": round(hit_rate, 2),
            "connected": self.client is not None,
        }

    def clear_all(self) -> bool:
        """Clear all cache entries (use with caution)."""
        if not self.client:
            return False

        try:
            self.client.flushdb()
            logger.warning("Cache CLEARED: All keys deleted from database")
            return True
        except RedisError as e:
            logger.error(f"Redis flush error: {e}")
            return False


def cache(
    namespace: str,
    key_func: Optional[Callable] = None,
    ttl: int = 300,
    cache_manager: Optional[CacheManager] = None,
):
    """
    Decorator for caching function results.

    Args:
        namespace: Cache namespace (e.g., "instruments", "prices")
        key_func: Function to generate cache key from args/kwargs
        ttl: Time to live in seconds
        cache_manager: CacheManager instance (uses global if None)

    Example:
        @cache(namespace="prices", ttl=60)
        async def get_latest_price(instrument_id: str):
            # ... expensive database query ...
            return price
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache manager
            mgr = cache_manager or _get_global_cache_manager()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: hash function name + args
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cached_value = mgr.get(namespace, cache_key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            mgr.set(namespace, cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


def cache_sync(
    namespace: str,
    key_func: Optional[Callable] = None,
    ttl: int = 300,
    cache_manager: Optional[CacheManager] = None,
):
    """Synchronous version of cache decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager
            mgr = cache_manager or _get_global_cache_manager()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: hash function name + args
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cached_value = mgr.get(namespace, cache_key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Store in cache
            mgr.set(namespace, cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def init_cache(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    default_ttl: int = 300,
):
    """Initialize global cache manager."""
    global _global_cache_manager
    _global_cache_manager = CacheManager(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        default_ttl=default_ttl,
    )
    return _global_cache_manager


def _get_global_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _global_cache_manager

    if _global_cache_manager is None:
        # Use environment variables or defaults
        import os
        _global_cache_manager = CacheManager(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "300")),
        )

    return _global_cache_manager


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    return _get_global_cache_manager()


# Cache invalidation helpers

def invalidate_instrument_cache(instrument_id: str):
    """Invalidate all cache entries for an instrument."""
    mgr = get_cache_manager()
    mgr.invalidate_pattern(f"instruments:*{instrument_id}*")
    mgr.invalidate_pattern(f"prices:*{instrument_id}*")
    mgr.invalidate_pattern(f"curves:*{instrument_id}*")


def invalidate_market_cache(market: str):
    """Invalidate all cache entries for a market."""
    mgr = get_cache_manager()
    mgr.invalidate_pattern(f"*:{market}:*")


def invalidate_all_caches():
    """Invalidate all cache entries (use with caution)."""
    mgr = get_cache_manager()
    mgr.clear_all()

