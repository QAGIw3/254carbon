"""Database utilities shared across services."""

from importlib import metadata

from .session import SessionFactory


__all__ = ["__version__", "SessionFactory"]


def __version__() -> str:
    try:
        return metadata.version("carbon254-libs-python")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0-dev"

