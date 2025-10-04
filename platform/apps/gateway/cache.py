"""
Intelligent Redis caching layer for API Gateway with adaptive TTL and cache warming.
"""
import logging
import json
import hashlib
import asyncio
from typing import Optional, Any, Dict, List, Callable
from datetime import timedelta, datetime
from enum import Enum

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different data types."""
    STATIC = "static"           # Static data, long TTL (1-24 hours)
    SEMI_STATIC = "semi_static"  # Semi-static data, medium TTL (15-60 minutes)
    DYNAMIC = "dynamic"         # Dynamic data, short TTL (1-5 minutes)
    REALTIME = "realtime"       # Real-time data, very short TTL (10-30 seconds)
    STREAMING = "streaming"     # Streaming data, no cache


class CacheManager:
    """Intelligent Redis caching manager with adaptive TTL and cache warming.

    Args:
        redis_url: Redis connection URL.
        default_ttl: Default TTL in seconds for unspecified strategies.
        max_connections: Connection pool size.
        retry_on_timeout: Retry flag for transient timeouts.
    """

    def __init__(
        self,
        redis_url: str = "redis://redis-cluster:6379",
        default_ttl: int = 300,  # 5 minutes
        max_connections: int = 20,
        retry_on_timeout: bool = True,
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self.retry_on_timeout = retry_on_timeout
        self._client: Optional[redis.Redis] = None
        self._pool: Optional[ConnectionPool] = None

        # Cache strategies configuration
        self.strategy_configs = {
            CacheStrategy.STATIC: {"ttl": 3600, "warm": True},        # 1 hour
            CacheStrategy.SEMI_STATIC: {"ttl": 900, "warm": True},   # 15 minutes
            CacheStrategy.DYNAMIC: {"ttl": 300, "warm": False},      # 5 minutes
            CacheStrategy.REALTIME: {"ttl": 30, "warm": False},      # 30 seconds
            CacheStrategy.STREAMING: {"ttl": 0, "warm": False},      # No cache
        }

        # Cache warming functions
        self.warm_functions: Dict[str, Callable] = {}

        # Performance tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "warm_count": 0,
        }
    
    async def get_client(self) -> redis.Redis:
        """Get or create Redis client with connection pooling.

        Returns:
            Redis client bound to a connection pool.
        """
        if self._client is None:
            # Create connection pool for better performance
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=self.retry_on_timeout,
                encoding="utf-8",
                decode_responses=True,
            )

            self._client = redis.Redis(
                connection_pool=self._pool,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
            )

        return self._client
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate a stable cache key from parameters.

        Args:
            prefix: Key prefix namespace.
            **kwargs: Parameters contributing to key identity.

        Returns:
            Stable hashed key string.
        """
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        params_str = json.dumps(sorted_params, sort_keys=True)
        
        # Hash parameters for shorter keys
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
        
        return f"254c:{prefix}:{params_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Parsed JSON value or None on miss/error.
        """
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
        """Set value in cache.

        Args:
            key: Cache key to set.
            value: Value to serialize as JSON.
            ttl: Time-to-live in seconds; defaults by strategy if None.

        Returns:
            True on success, False on error.
        """
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
    
    def get_ttl_for_strategy(self, strategy: CacheStrategy) -> int:
        """Get TTL for a given cache strategy."""
        return self.strategy_configs[strategy]["ttl"]

    async def get_with_strategy(
        self,
        key: str,
        strategy: CacheStrategy = CacheStrategy.DYNAMIC
    ) -> Optional[Any]:
        """Get value with strategy-based TTL."""
        try:
            client = await self.get_client()
            value = await client.get(key)

            if value:
                self._stats["hits"] += 1
                logger.debug(f"Cache HIT: {key} (strategy: {strategy.value})")
                return json.loads(value)
            else:
                self._stats["misses"] += 1
                logger.debug(f"Cache MISS: {key} (strategy: {strategy.value})")
                return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None

    async def set_with_strategy(
        self,
        key: str,
        value: Any,
        strategy: CacheStrategy = CacheStrategy.DYNAMIC,
    ) -> bool:
        """Set value with strategy-based TTL."""
        try:
            client = await self.get_client()
            ttl = self.get_ttl_for_strategy(strategy)

            if ttl == 0:  # No caching for streaming data
                return True

            serialized = json.dumps(value)
            await client.setex(key, timedelta(seconds=ttl), serialized)

            logger.debug(f"Cache SET: {key} (TTL={ttl}s, strategy={strategy.value})")
            return True
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False

    def register_warm_function(self, key_prefix: str, warm_func: Callable):
        """Register a function to warm cache for a key pattern."""
        self.warm_functions[key_prefix] = warm_func

    async def warm_cache(self, key_prefix: str) -> bool:
        """Warm cache for a specific key pattern."""
        if key_prefix not in self.warm_functions:
            logger.warning(f"No warm function registered for prefix: {key_prefix}")
            return False

        try:
            warm_func = self.warm_functions[key_prefix]
            result = await warm_func()

            # Cache the result
            await self.set_with_strategy(
                f"warm:{key_prefix}",
                result,
                CacheStrategy.STATIC
            )

            self._stats["warm_count"] += 1
            logger.info(f"Cache warmed for prefix: {key_prefix}")
            return True
        except Exception as e:
            logger.error(f"Cache warming failed for {key_prefix}: {e}")
            return False

    async def warm_all_cache(self) -> Dict[str, bool]:
        """Warm all registered cache patterns."""
        results = {}
        for prefix in self.warm_functions.keys():
            results[prefix] = await self.warm_cache(prefix)

        logger.info(f"Cache warming completed: {sum(results.values())}/{len(results)} successful")
        return results

    async def get_stats(self) -> dict:
        """Get comprehensive cache statistics."""
        try:
            client = await self.get_client()
            info = await client.info("stats")

            # Combine Redis stats with our custom stats
            redis_stats = {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "total_keys": await client.dbsize(),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients", 0),
                "evicted_keys": info.get("evicted_keys", 0),
            }

            # Calculate hit rate
            total_requests = redis_stats["hits"] + redis_stats["misses"]
            hit_rate = (redis_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                **redis_stats,
                "hit_rate": hit_rate,
                "custom_stats": self._stats.copy(),
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}


# Cache decorators

def cache_response(
    prefix: str,
    strategy: CacheStrategy = CacheStrategy.DYNAMIC,
    key_func: Optional[callable] = None,
):
    """
    Decorator to cache endpoint responses with intelligent strategy.

    Usage:
        @cache_response("instruments", CacheStrategy.SEMI_STATIC)
        async def get_instruments(market: str):
            ...

        @cache_response("realtime_prices", CacheStrategy.REALTIME)
        async def get_realtime_prices(node_id: str):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._generate_key(prefix, **kwargs)

            # Try to get from cache with strategy
            cached_value = await cache_manager.get_with_strategy(cache_key, strategy)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache with strategy
            await cache_manager.set_with_strategy(cache_key, result, strategy)

            return result

        return wrapper
    return decorator


def cache_invalidate(pattern: str):
    """
    Decorator to invalidate cache patterns after function execution.

    Usage:
        @cache_invalidate("instruments:*")
        async def update_instruments():
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Invalidate cache pattern
            await cache_manager.delete_pattern(pattern)

            return result

        return wrapper
    return decorator


# Cache warming functions for common endpoints

async def warm_instruments_cache():
    """Warm cache for instrument data."""
    import db

    try:
        pool = await db.get_postgres_pool()
        async with pool.acquire() as conn:
            # Get all instruments for cache warming
            instruments = await conn.fetch("""
                SELECT instrument_id, name, market, product_type
                FROM pg.instrument
                WHERE active = true
                ORDER BY market, instrument_id
            """)

            return {
                "instruments": [dict(row) for row in instruments],
                "total_count": len(instruments),
                "last_updated": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to warm instruments cache: {e}")
        return {"error": str(e)}


async def warm_markets_cache():
    """Warm cache for market data."""
    from db import get_postgres_pool

    try:
        pool = await db.get_postgres_pool()
        async with pool.acquire() as conn:
            # Get market metadata
            markets = await conn.fetch("""
                SELECT DISTINCT market, description, timezone, status
                FROM pg.instrument
                WHERE active = true
                GROUP BY market, description, timezone, status
                ORDER BY market
            """)

            return {
                "markets": [dict(row) for row in markets],
                "total_count": len(markets),
                "last_updated": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to warm markets cache: {e}")
        return {"error": str(e)}


async def warm_curve_metadata_cache():
    """Warm cache for curve metadata."""
    from db import get_postgres_pool

    try:
        pool = await db.get_postgres_pool()
        async with pool.acquire() as conn:
            # Get curve metadata for cache warming
            curves = await conn.fetch("""
                SELECT DISTINCT curve_id, market, product, description
                FROM pg.forward_curve_points
                WHERE as_of_date >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY curve_id, market, product, description
                ORDER BY market, curve_id
            """)

            return {
                "curves": [dict(row) for row in curves],
                "total_count": len(curves),
                "last_updated": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to warm curve metadata cache: {e}")
        return {"error": str(e)}


# Register cache warming functions
def initialize_cache_warming():
    """Initialize cache warming for common endpoints."""
    cache_manager.register_warm_function("instruments", warm_instruments_cache)
    cache_manager.register_warm_function("markets", warm_markets_cache)
    cache_manager.register_warm_function("curves", warm_curve_metadata_cache)

    logger.info("Cache warming functions registered")


# Initialize cache warming on module import
initialize_cache_warming()


# Global cache manager instance with enhanced configuration
cache_manager = CacheManager(
    redis_url="redis://redis-cluster:6379",
    max_connections=20,
    retry_on_timeout=True,
)
