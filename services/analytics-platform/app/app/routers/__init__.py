"""API routers for the analytics platform."""

from .legacy import router as legacy_router
from .v1 import router as v1_router

__all__ = ["legacy_router", "v1_router"]

