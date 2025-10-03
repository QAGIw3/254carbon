"""
Redis caching layer for API Gateway.
"""
import logging
import json
import hashlib
from typing import Optional, Any
from datetime import timedelta

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheManager:
    """Manage Redis caching for API responses."""
    
    def __init__(
        self,
        redis_url: str = "redis://redis-cluster:6379",
        default_ttl: int = 300,  # 5 minutes
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._client: Optional[redis.Redis] = None
    
    async def get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        params_str = json.dumps(sorted_params, sort_keys=True)
        
        # Hash parameters for shorter keys
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
        
        return f"254c:{prefix}:{params_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            client = await self.get_client()
            value = await client.get(key)
            
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            else:
                logger.debug(f"Cache MISS: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache."""
        try:
            client = await self.get_client()
            
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            
            await client.setex(
                key,
                timedelta(seconds=ttl),
                serialized,
            )
            
            logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            client = await self.get_client()
            await client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            client = await self.get_client()
            keys = await client.keys(pattern)
            
            if keys:
                deleted = await client.delete(*keys)
                logger.info(f"Cache DELETE pattern '{pattern}': {deleted} keys")
                return deleted
            
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
            return 0
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            client = await self.get_client()
            info = await client.info("stats")
            
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) /
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                ) * 100,
                "total_keys": await client.dbsize(),
                "used_memory_human": info.get("used_memory_human"),
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}


# Cache decorators

def cache_response(
    prefix: str,
    ttl: int = 300,
    key_func: Optional[callable] = None,
):
    """
    Decorator to cache endpoint responses.
    
    Usage:
        @cache_response("instruments", ttl=600)
        async def get_instruments(market: str):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._generate_key(prefix, **kwargs)
            
            # Try to get from cache
            cached_value = await cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
cache_manager = CacheManager()

