"""HTTP utilities shared across services."""

from importlib import metadata

from .middleware import RequestTimingMiddleware
from .rate_limit import TokenBucket


__all__ = ["__version__", "RequestTimingMiddleware", "TokenBucket"]


def __version__() -> str:
    try:
        return metadata.version("carbon254-libs-python")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0-dev"

