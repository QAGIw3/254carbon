"""FastAPI router exposing commodity research analytics endpoints."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from commodity_research_framework import CommodityResearchFramework
from data_access import DataAccessLayer
from research_config import supply_demand_mapping, weather_mapping
from research_persistence import ResearchPersistence
from ui_hooks import (
    publish_decomposition_update,
    publish_supply_demand_update,
    publish_volatility_update,
    publish_weather_impact_update,
)


logger = logging.getLogger(__name__)

router = APIRouter()

_data_access = DataAccessLayer()
_persistence = ResearchPersistence(ch_client=_data_access.client)
_framework = CommodityResearchFramework(
    data_access=_data_access,
    persistence=_persistence,
)

try:  # Prometheus metrics are optional
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - metrics optionality
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

if Counter and Histogram:
    _API_CALL_COUNTER = Counter(
        "ml_service_research_api_requests_total",
        "Total research API requests served",
        labelnames=("endpoint",),
    )
    _API_LATENCY = Histogram(
        "ml_service_research_api_latency_seconds",
        "Latency per research API endpoint",
        labelnames=("endpoint",),
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )
else:  # pragma: no cover - metrics disabled path
    _API_CALL_COUNTER = None
    _API_LATENCY = None


def _instrument(endpoint: str):
    def decorator(func):
        if Counter is None or Histogram is None:
            return func

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if _API_CALL_COUNTER is not None:
                _API_CALL_COUNTER.labels(endpoint=endpoint).inc()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                if _API_LATENCY is not None:
                    duration = time.perf_counter() - start
                    _API_LATENCY.labels(endpoint=endpoint).observe(duration)

        return wrapper

    return decorator


def _series_to_points(series: pd.Series) -> List[Dict[str, Any]]:
    series = series.dropna()
    return [
        {"date": pd.to_datetime(idx).date(), "value": float(val)}
        for idx, val in series.sort_index().items()
    ]


class DecompositionRequest(BaseModel):
    instrument_id: str = Field(..., description="Instrument identifier")
    commodity_type: str = Field(..., description="Commodity type (e.g., gas, power, oil)")
    method: str = Field("stl", description="Decomposition method", regex="^(stl|classical)$")
    start_date: Optional[datetime] = Field(None, description="Optional start date filter")
    end_date: Optional[datetime] = Field(None, description="Optional end date filter")
    persist: bool = Field(True, description="Persist results to ClickHouse")
    version: str = Field("v1", description="Version tag for persistence")
    publish: bool = Field(False, description="Publish results to Web Hub")


class DecompositionPoint(BaseModel):
    date: datetime
    trend: Optional[float]
    seasonal: Optional[float]
    residual: Optional[float]


class DecompositionResponse(BaseModel):
    instrument_id: str
    method: str
    snapshot_date: datetime
    components: List[DecompositionPoint]
    metadata: Dict[str, Any]


class VolatilityRegimeResponse(BaseModel):
    instrument_id: str
    method: str
    n_regimes: int
    as_of: datetime
    regimes: Dict[str, Dict[str, Optional[float]]]
    labels: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SupplyDemandResponse(BaseModel):
    instrument_id: str
    entity_id: str
    as_of: datetime
    metrics: Dict[str, List[Dict[str, Any]]]
    latest: Dict[str, Optional[float]]
    units: Dict[str, str]


class WeatherImpactCoefficient(BaseModel):
    coef_type: str
    coefficient: Optional[float]
    p_value: Optional[float]


class WeatherImpactResponse(BaseModel):
    entity_id: str
    as_of: datetime
    method: str
    r_squared: Optional[float]
    window: str
    model_version: str
    extreme_event_count: int
    coefficients: List[WeatherImpactCoefficient]
    diagnostics: Dict[str, Any]


@router.post("/api/v1/research/decomposition", response_model=DecompositionResponse)
@_instrument("decomposition")
async def compute_decomposition(request: DecompositionRequest) -> DecompositionResponse:
    try:
        result = _framework.generate_time_series_decomposition(
            instrument_id=request.instrument_id,
            commodity_type=request.commodity_type,
            method=request.method,
            start=request.start_date,
            end=request.end_date,
            version=request.version,
            persist=request.persist,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Decomposition failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    df = pd.concat(
        [
            result.components.get("trend", pd.Series(dtype=float)).rename("trend"),
            result.components.get("seasonal", pd.Series(dtype=float)).rename("seasonal"),
            result.components.get("residual", pd.Series(dtype=float)).rename("residual"),
        ],
        axis=1,
    ).dropna(how="all").sort_index()

    points = [
        DecompositionPoint(
            date=pd.to_datetime(idx),
            trend=_safe_value(row.get("trend")),
            seasonal=_safe_value(row.get("seasonal")),
            residual=_safe_value(row.get("residual")),
        )
        for idx, row in df.iterrows()
    ]

    response = DecompositionResponse(
        instrument_id=result.instrument_id,
        method=result.method,
        snapshot_date=result.snapshot_date,
        components=points,
        metadata=result.metadata,
    )

    if request.publish:
        publish_decomposition_update(result)

    return response


def _safe_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


@router.get("/api/v1/research/volatility-regimes", response_model=VolatilityRegimeResponse)
@_instrument("volatility_regimes")
async def get_volatility_regimes(
    instrument_id: str = Query(..., description="Instrument identifier"),
    method: str = Query("auto", description="Method override (auto|kmeans|markov|hmm)"),
    n_regimes: int = Query(3, ge=2, le=6, description="Number of regimes"),
    lookback_days: int = Query(365, ge=30, description="Lookback window for returns"),
    persist: bool = Query(False, description="Persist regime assignments"),
    publish: bool = Query(False, description="Publish results to Web Hub"),
) -> VolatilityRegimeResponse:
    try:
        result = _framework.analyze_volatility_regimes(
            instrument_id=instrument_id,
            method=method,
            n_regimes=n_regimes,
            lookback_days=lookback_days,
            persist=persist,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Volatility regime detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    labels = [
        {"date": pd.to_datetime(idx).date(), "label": label}
        for idx, label in result.labels.sort_index().items()
    ]

    regimes: Dict[str, Dict[str, Optional[float]]] = {}
    for regime, stats_dict in result.regime_profiles.items():
        regimes[regime] = {
            key: _safe_value(val) for key, val in stats_dict.items()
        }

    response = VolatilityRegimeResponse(
        instrument_id=result.instrument_id,
        method=result.method,
        n_regimes=result.n_regimes,
        as_of=result.as_of,
        regimes=regimes,
        labels=labels,
        metadata=result.metadata,
    )

    if publish:
        publish_volatility_update(result)

    return response


@router.get("/api/v1/research/sd-balance", response_model=SupplyDemandResponse)
@_instrument("supply_demand_balance")
async def get_supply_demand_balance(
    instrument_id: str = Query(..., description="Commodity instrument id"),
    entity_id: Optional[str] = Query(None, description="Entity or region identifier"),
    inventory_entity_id: Optional[str] = Query(None, description="Inventory entity id"),
    inventory_variable: Optional[str] = Query(None, description="Inventory variable"),
    production_entity_id: Optional[str] = Query(None, description="Production entity id"),
    production_variable: Optional[str] = Query(None, description="Production variable"),
    consumption_entity_id: Optional[str] = Query(None, description="Consumption entity id"),
    consumption_variable: Optional[str] = Query(None, description="Consumption variable"),
    price_lookback_days: int = Query(365, ge=90, description="Price lookback window"),
    fundamentals_lookback_days: int = Query(365, ge=90, description="Fundamentals lookback window"),
    persist: bool = Query(False, description="Persist results"),
    publish: bool = Query(False, description="Publish results to Web Hub"),
) -> SupplyDemandResponse:
    mapping = supply_demand_mapping(instrument_id)
    resolved_entity_id = entity_id or mapping.get("entity_id") or instrument_id

    inventory_cfg = mapping.get("inventory", {})
    production_cfg = mapping.get("production", {})
    consumption_cfg = mapping.get("consumption", {})

    inventory_entity_id = inventory_entity_id or inventory_cfg.get("entity_id")
    inventory_variable = inventory_variable or inventory_cfg.get("variable")
    production_entity_id = production_entity_id or production_cfg.get("entity_id")
    production_variable = production_variable or production_cfg.get("variable")
    consumption_entity_id = consumption_entity_id or consumption_cfg.get("entity_id")
    consumption_variable = consumption_variable or consumption_cfg.get("variable")

    if not any([inventory_entity_id, production_entity_id, consumption_entity_id]):
        raise HTTPException(
            status_code=400,
            detail="No supply/demand mapping provided; configure config or query parameters",
        )

    for ent, var, label in (
        (inventory_entity_id, inventory_variable, "inventory"),
        (production_entity_id, production_variable, "production"),
        (consumption_entity_id, consumption_variable, "consumption"),
    ):
        if (ent and not var) or (var and not ent):
            raise HTTPException(
                status_code=400,
                detail=f"{label} mapping requires both entity_id and variable",
            )

    price_lookback_days = mapping.get("lookback_days", price_lookback_days)
    prices = _framework.data_access.get_price_series(
        instrument_id,
        lookback_days=price_lookback_days,
    )
    if prices.empty:
        raise HTTPException(status_code=404, detail="No price history available")

    inventory_series = None
    if inventory_entity_id and inventory_variable:
        inventory_series = _framework.data_access.get_fundamental_series(
            entity_id=inventory_entity_id,
            variable=inventory_variable,
            lookback_days=inventory_cfg.get("lookback_days", fundamentals_lookback_days),
        )
        unit_override = inventory_cfg.get("unit")
        if unit_override:
            inventory_series.attrs["unit"] = unit_override

    production_series = None
    if production_entity_id and production_variable:
        production_series = _framework.data_access.get_fundamental_series(
            entity_id=production_entity_id,
            variable=production_variable,
            lookback_days=production_cfg.get("lookback_days", fundamentals_lookback_days),
        )
        unit_override = production_cfg.get("unit")
        if unit_override:
            production_series.attrs["unit"] = unit_override

    consumption_series = None
    if consumption_entity_id and consumption_variable:
        consumption_series = _framework.data_access.get_fundamental_series(
            entity_id=consumption_entity_id,
            variable=consumption_variable,
            lookback_days=consumption_cfg.get("lookback_days", fundamentals_lookback_days),
        )
        unit_override = consumption_cfg.get("unit")
        if unit_override:
            consumption_series.attrs["unit"] = unit_override

    result = _framework.model_supply_demand_balance(
        prices=prices,
        inventory_data=inventory_series,
        production_data=production_series,
        consumption_data=consumption_series,
        instrument_id=instrument_id,
        entity_id=resolved_entity_id,
        persist=persist,
    )

    metrics_payload: Dict[str, List[Dict[str, Any]]] = {}
    latest_metrics: Dict[str, Optional[float]] = {}
    for metric_name, series in result.metrics.items():
        if series is None or series.empty:
            metrics_payload[metric_name] = []
            latest_metrics[metric_name] = None
            continue
        cleaned = series.dropna()
        metrics_payload[metric_name] = _series_to_points(cleaned)
        latest_metrics[metric_name] = _safe_value(cleaned.sort_index().iloc[-1]) if not cleaned.empty else None

    response = SupplyDemandResponse(
        instrument_id=instrument_id,
        entity_id=result.entity_id,
        as_of=result.as_of,
        metrics=metrics_payload,
        latest=latest_metrics,
        units=result.units,
    )

    if publish:
        publish_supply_demand_update(result)

    return response


@router.get("/api/v1/research/weather-impact", response_model=WeatherImpactResponse)
@_instrument("weather_impact")
async def get_weather_impact(
    instrument_id: Optional[str] = Query(None, description="Instrument id for price series"),
    entity_id: Optional[str] = Query(None, description="Entity identifier for weather mapping"),
    temperature_entity_id: Optional[str] = Query(None, description="Entity id for temperature series"),
    temperature_variable: Optional[str] = Query(None, description="Variable name for temperature series"),
    hdd_entity_id: Optional[str] = Query(None, description="Entity id for heating degree days"),
    hdd_variable: Optional[str] = Query(None, description="Variable name for HDD"),
    cdd_entity_id: Optional[str] = Query(None, description="Entity id for cooling degree days"),
    cdd_variable: Optional[str] = Query(None, description="Variable name for CDD"),
    lookback_days: int = Query(365, ge=90, description="Lookback window"),
    window_days: int = Query(90, ge=30, description="Rolling diagnostic window"),
    lags: Optional[str] = Query(None, description="Comma-separated temperature lags"),
    persist: bool = Query(False, description="Persist regression output"),
    publish: bool = Query(False, description="Publish results to Web Hub"),
) -> WeatherImpactResponse:
    mapping = weather_mapping(entity_id or instrument_id or "")

    instrument_id = instrument_id or mapping.get("instrument_id")
    if not instrument_id:
        raise HTTPException(status_code=400, detail="instrument_id is required")

    temperature_cfg = mapping.get("temperature", {})
    temperature_entity_id = temperature_entity_id or temperature_cfg.get("entity_id")
    temperature_variable = temperature_variable or temperature_cfg.get("variable")
    if not temperature_entity_id or not temperature_variable:
        raise HTTPException(status_code=400, detail="temperature mapping requires entity_id and variable")

    hdd_cfg = mapping.get("hdd", {})
    cdd_cfg = mapping.get("cdd", {})
    hdd_entity_id = hdd_entity_id or hdd_cfg.get("entity_id")
    hdd_variable = hdd_variable or hdd_cfg.get("variable")
    cdd_entity_id = cdd_entity_id or cdd_cfg.get("entity_id")
    cdd_variable = cdd_variable or cdd_cfg.get("variable")

    if (hdd_entity_id and not hdd_variable) or (hdd_variable and not hdd_entity_id):
        raise HTTPException(status_code=400, detail="hdd mapping requires both entity_id and variable")
    if (cdd_entity_id and not cdd_variable) or (cdd_variable and not cdd_entity_id):
        raise HTTPException(status_code=400, detail="cdd mapping requires both entity_id and variable")

    entity_id = entity_id or mapping.get("entity_id") or temperature_entity_id
    lookback_days = mapping.get("lookback_days", lookback_days)
    window_days = mapping.get("window_days", window_days)

    prices = _framework.data_access.get_price_series(
        instrument_id,
        lookback_days=lookback_days,
    )
    if prices.empty:
        raise HTTPException(status_code=404, detail="No price history available")

    temperature_series = _framework.data_access.get_weather_series(
        temperature_entity_id,
        temperature_variable,
        lookback_days=lookback_days,
    )
    if temperature_series.empty:
        raise HTTPException(status_code=404, detail="Temperature series unavailable")

    heating_series = None
    if hdd_entity_id and hdd_variable:
        heating_series = _framework.data_access.get_fundamental_series(
            entity_id=hdd_entity_id,
            variable=hdd_variable,
            lookback_days=lookback_days,
        )

    cooling_series = None
    if cdd_entity_id and cdd_variable:
        cooling_series = _framework.data_access.get_fundamental_series(
            entity_id=cdd_entity_id,
            variable=cdd_variable,
            lookback_days=lookback_days,
        )

    lag_values: Optional[List[int]] = None
    if lags:
        try:
            lag_values = [int(item.strip()) for item in lags.split(",") if item.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid lags parameter") from exc
    elif isinstance(mapping.get("lags"), list):
        lag_values = [int(val) for val in mapping.get("lags")]

    try:
        result = _framework.analyze_weather_impact(
            prices=prices,
            temperature_data=temperature_series,
            heating_demand=heating_series,
            cooling_demand=cooling_series,
            entity_id=entity_id,
            window=f"{window_days}D",
            lags=lag_values,
            persist=persist,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Weather impact calibration failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    coefficients = [
        WeatherImpactCoefficient(
            coef_type=coef_type,
            coefficient=_safe_value(stats.get("coef")),
            p_value=_safe_value(stats.get("p_value")),
        )
        for coef_type, stats in result.coefficients.items()
    ]

    response = WeatherImpactResponse(
        entity_id=result.entity_id,
        as_of=result.as_of,
        method=result.method,
        r_squared=_safe_value(result.r_squared),
        window=result.window,
        model_version=result.model_version,
        extreme_event_count=result.extreme_event_count,
        coefficients=coefficients,
        diagnostics=result.diagnostics,
    )

    if publish:
        publish_weather_impact_update(result)

    return response
