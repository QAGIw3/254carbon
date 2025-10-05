"""Cache utilities for edge gateway."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from cachetools import TTLCache


class CacheStrategy:
    SEMI_STATIC = "semi_static"
    DYNAMIC = "dynamic"


def cache_response(prefix: str, strategy: str):
    def decorator(func: Callable):
        return func

    return decorator

