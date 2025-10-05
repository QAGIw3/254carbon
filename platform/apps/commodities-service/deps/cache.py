"""
Caching helpers for the commodities service.

This module provides a thin decorator around the platform-shared Redis
CacheManager. It standardizes TTL tiers for endpoints and generates stable
cache keys based on request parameters.

Highlights
- Uses `jsonable_encoder` to ensure Pydantic models/lists are safely cached.
- Keeps key generation deterministic via normalized payload hashing.
- Reads Redis connection details from environment, falling back to sensible
  defaults for local development.
"""
import hashlib
import json
import logging
import os
import sys
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastapi.encoders import jsonable_encoder

_PLATFORM_DIR = Path(__file__).resolve().parents[3]
if str(_PLATFORM_DIR) not in sys.path and _PLATFORM_DIR.exists():
    sys.path.insert(0, str(_PLATFORM_DIR))

from shared.cache_utils import CacheManager  # type: ignore

logger = logging.getLogger(__name__)


class CacheTTL(int, Enum):
    """Standard TTL buckets for API responses (seconds)."""

    REALTIME = 45  # 30–60s window (hub prices, emissions)
    SEMI_STATIC = 600  # 5–15m window (curves, analytics)
    STATIC = 3600  # 1–6h window (metadata/specs)


def _serialize_for_cache(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """
    Serialize function arguments into a stable SHA-256 key digest.

    We avoid leaking PII or raw payloads by hashing the normalized structure
    instead of embedding full query strings.
    """

    def normalize(value: Any) -> Any:
        if hasattr(value, "dict"):
            return value.dict()  # Pydantic/BaseModel support
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, dict):
            return {k: normalize(value[k]) for k in sorted(value)}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    payload = {
        "args": [normalize(arg) for arg in args],
        "kwargs": {k: normalize(v) for k, v in sorted(kwargs.items())},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_cache_manager() -> CacheManager:
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    prefix = os.getenv("CACHE_PREFIX", "commodities-service")
    default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "300"))

    # Construct a single CacheManager instance. The shared utility will disable
    # itself gracefully if Redis is unreachable, which maintains endpoint
    # correctness (only losing caching benefits).
    return CacheManager(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        default_ttl=default_ttl,
        prefix=prefix,
    )


_cache_manager: CacheManager = _build_cache_manager()


def get_cache_manager() -> CacheManager:
    """Expose the cache manager for diagnostics."""

    return _cache_manager


def cache_response(
    namespace: str,
    ttl: CacheTTL | int,
    key_builder: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Cache decorator leveraging the shared CacheManager.

    Usage
    -----
    @router.get("/path")
    @cache_response("namespace", CacheTTL.SEMI_STATIC, key_builder=my_key)
    async def handler(...):
        ...
    """

    ttl_seconds = int(ttl)

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = (
                key_builder(*args, **kwargs)
                if key_builder
                else _serialize_for_cache(args, kwargs)
            )

            # We pass a per-namespace key; CacheManager prefixes with service
            # namespace to avoid collisions across apps.
            namespaced_key = cache_key
            cached = _cache_manager.get(namespace, namespaced_key)
            if cached is not None:
                logger.debug("Cache hit for %s", namespaced_key)
                return cached

            # Execute and cache the serializable form of the result. We still
            # return the original object for convenience.
            result = await func(*args, **kwargs)
            serializable = jsonable_encoder(result)
            try:
                _cache_manager.set(namespace, namespaced_key, serializable, ttl=ttl_seconds)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to cache response for %s: %s", namespaced_key, exc)
            return result

        return wrapper

    return decorator


__all__ = [
    "CacheTTL",
    "cache_response",
    "get_cache_manager",
]
