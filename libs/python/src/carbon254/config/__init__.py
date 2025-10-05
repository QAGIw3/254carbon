"""Configuration helpers for 12-factor services."""

from importlib import metadata

from .settings import ServiceSettings, load_settings


__all__ = ["__version__", "ServiceSettings", "load_settings"]


def __version__() -> str:
    try:
        return metadata.version("carbon254-libs-python")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0-dev"

