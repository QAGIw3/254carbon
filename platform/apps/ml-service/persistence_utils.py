"""Shared helpers for writing analytics outputs to ClickHouse."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Optional


logger = logging.getLogger(__name__)


def as_date(value: Any, *, field: str = "as_of_date") -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if value is None:
        raise ValueError(f"{field} is required for persistence")
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(float(value)).date()
    text = str(value)
    for fmt in (None, "%Y-%m-%d", "%Y/%m/%d"):
        try:
            if fmt is None:
                return datetime.fromisoformat(text).date()
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date value for {field}: {value!r}")


def json_dump(value: Any, *, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        dumped = json.dumps(value, default=str)
    except (TypeError, ValueError):
        logger.warning("Failed to encode payload as JSON", exc_info=True)
        return default
    return dumped


def json_text(value: Any, *, default: str = "{}") -> str:
    dumped = json_dump(value)
    if dumped in (None, "null", "None"):
        return default
    return dumped


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Failed to coerce value %r to float", value, exc_info=True)
        return default


def optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Failed to coerce value %r to optional float", value, exc_info=True)
        return None


def to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Failed to coerce value %r to int", value, exc_info=True)
        return default


__all__ = [
    "as_date",
    "json_dump",
    "json_text",
    "optional_float",
    "to_float",
    "to_int",
]

