"""Logging and observability helpers."""

from importlib import metadata

from .structured import JsonLogFormatter, configure_json_logging


__all__ = ["__version__", "JsonLogFormatter", "configure_json_logging"]


def __version__() -> str:
    try:
        return metadata.version("carbon254-libs-python")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0-dev"

