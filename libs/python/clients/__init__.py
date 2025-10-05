"""Shared client interfaces and base implementations."""

from importlib import metadata

from .base import BaseHttpClient, CircuitOpenError


__all__ = ["__version__", "BaseHttpClient", "CircuitOpenError"]


def __version__() -> str:
    """Return the package version if installed."""

    try:
        return metadata.version("carbon254-libs-python")
    except metadata.PackageNotFoundError:  # pragma: no cover - during development
        return "0.0.0-dev"

