"""Helpers for loading commodity research configuration mappings."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "commodity_research.yaml"


@lru_cache(maxsize=1)
def load_research_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists():
        logger.warning("Research config not found at %s", _CONFIG_PATH)
        return {}

    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - config error fallback
        logger.error("Failed to load research config: %s", exc)
        return {}
    return data


def get_instrument_mapping(instrument_id: str) -> Dict[str, Any]:
    config = load_research_config()
    return config.get("instruments", {}).get(instrument_id, {})


def get_entity_mapping(entity_id: str) -> Dict[str, Any]:
    config = load_research_config()
    return config.get("entities", {}).get(entity_id, {})


def supply_demand_mapping(instrument_id: str) -> Dict[str, Any]:
    mapping = get_instrument_mapping(instrument_id)
    return mapping.get("supply_demand", {})


def weather_mapping(entity_id: str) -> Dict[str, Any]:
    mapping = get_entity_mapping(entity_id)
    if mapping:
        return mapping.get("weather", {})
    config = load_research_config()
    return config.get("default_weather", {})


__all__ = [
    "load_research_config",
    "get_instrument_mapping",
    "get_entity_mapping",
    "supply_demand_mapping",
    "weather_mapping",
]

