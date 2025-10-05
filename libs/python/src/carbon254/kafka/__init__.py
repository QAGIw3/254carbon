"""Kafka utilities for producers and consumers."""

from importlib import metadata

from .producer import BaseKafkaProducer, SchemaValidationError


__all__ = ["__version__", "BaseKafkaProducer", "SchemaValidationError"]


def __version__() -> str:
    try:
        return metadata.version("carbon254-libs-python")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0-dev"

