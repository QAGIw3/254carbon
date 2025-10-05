"""Base Kafka producer with schema validation."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict

from confluent_kafka import Producer


class SchemaValidationError(ValueError):
    """Raised when payload validation fails."""


class BaseKafkaProducer:
    def __init__(
        self,
        config: Dict[str, Any],
        schema_validator: Callable[[Dict[str, Any]], None],
    ) -> None:
        self._producer = Producer(config)
        self._schema_validator = schema_validator

    def produce(self, topic: str, payload: Dict[str, Any]) -> None:
        self._schema_validator(payload)
        self._producer.produce(topic, json.dumps(payload).encode("utf-8"))

    def flush(self, timeout: float | None = None) -> None:
        self._producer.flush(timeout)


__all__ = ["BaseKafkaProducer", "SchemaValidationError"]

