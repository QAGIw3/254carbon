"""Publish commodity research results to Web Hub endpoints."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests

from commodity_research_framework import (
    DecompositionResult,
    SupplyDemandResult,
    VolatilityRegimeResult,
    WeatherImpactResult,
)


logger = logging.getLogger(__name__)

WEB_HUB_ENDPOINT = os.getenv("WEB_HUB_ENDPOINT")


def _post(path: str, payload: Dict[str, Any]) -> bool:
    if not WEB_HUB_ENDPOINT:
        logger.debug("WEB_HUB_ENDPOINT not configured; skipping UI publish")
        return False

    url = f"{WEB_HUB_ENDPOINT.rstrip('/')}/{path.lstrip('/')}"
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return True
    except Exception:  # pragma: no cover - best effort
        logger.warning("Failed to publish UI payload to %s", url, exc_info=True)
        return False


def _series_to_points(series: pd.Series) -> Iterable[Dict[str, Any]]:
    series = series.dropna()
    for idx, value in series.sort_index().items():
        yield {"date": pd.to_datetime(idx).isoformat(), "value": float(value)}


def publish_decomposition_update(result: DecompositionResult) -> bool:
    components_df = pd.concat(
        [
            result.components.get("trend", pd.Series(dtype=float)).rename("trend"),
            result.components.get("seasonal", pd.Series(dtype=float)).rename("seasonal"),
            result.components.get("residual", pd.Series(dtype=float)).rename("residual"),
        ],
        axis=1,
    ).dropna(how="all").sort_index()

    payload = {
        "instrument_id": result.instrument_id,
        "commodity_type": result.commodity_type,
        "method": result.method,
        "snapshot_date": result.snapshot_date.isoformat(),
        "version": result.version,
        "metadata": result.metadata,
        "points": [
            {
                "date": pd.to_datetime(idx).isoformat(),
                "trend": row.get("trend"),
                "seasonal": row.get("seasonal"),
                "residual": row.get("residual"),
            }
            for idx, row in components_df.iterrows()
        ],
    }

    return _post("api/research/decomposition", payload)


def publish_volatility_update(result: VolatilityRegimeResult) -> bool:
    payload = {
        "instrument_id": result.instrument_id,
        "method": result.method,
        "n_regimes": result.n_regimes,
        "as_of": result.as_of.isoformat(),
        "regimes": result.regime_profiles,
        "labels": [
            {"date": pd.to_datetime(idx).isoformat(), "label": label}
            for idx, label in result.labels.sort_index().items()
        ],
        "metadata": result.metadata,
    }
    return _post("api/research/volatility", payload)


def publish_supply_demand_update(result: SupplyDemandResult) -> bool:
    payload = {
        "entity_id": result.entity_id,
        "instrument_id": result.instrument_id,
        "as_of": result.as_of.isoformat(),
        "version": result.version,
        "metrics": {
            metric: list(_series_to_points(series)) if series is not None else []
            for metric, series in result.metrics.items()
        },
        "units": result.units,
        "metadata": result.metadata,
    }
    return _post("api/research/supply-demand", payload)


def publish_weather_impact_update(result: WeatherImpactResult) -> bool:
    payload = {
        "entity_id": result.entity_id,
        "as_of": result.as_of.isoformat(),
        "method": result.method,
        "r_squared": result.r_squared,
        "window": result.window,
        "model_version": result.model_version,
        "extreme_event_count": result.extreme_event_count,
        "coefficients": result.coefficients,
        "diagnostics": result.diagnostics,
    }
    return _post("api/research/weather-impact", payload)


__all__ = [
    "publish_decomposition_update",
    "publish_volatility_update",
    "publish_supply_demand_update",
    "publish_weather_impact_update",
]

