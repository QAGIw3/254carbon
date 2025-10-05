"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


class JsonLogFormatter(logging.Formatter):
    """Format logs as JSON for ingestion by observability pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "message": record.getMessage(),
            "level": record.levelname,
            "logger": record.name,
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }

        if record.exc_info:
            log_payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_payload["stack"] = self.formatStack(record.stack_info)

        # Attach extra fields (excluding reserved attributes).
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in logging.LogRecord.__dict__:
                continue
            log_payload[key] = value

        return json.dumps(log_payload, default=str)


def configure_json_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


__all__ = ["configure_json_logging", "JsonLogFormatter"]

