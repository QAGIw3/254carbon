"""FastAPI router exposing renewables analytics endpoints."""

from __future__ import annotations

import logging
import time
from datetime import date
from functools import wraps
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from biodiesel_spread_analysis import BiodieselSpreadAnalysis
from carbon_intensity_calculator import CarbonIntensityCalculator
from data_access import DataAccessLayer
from renewables_persistence import RenewablesPersistence
from rin_price_forecast import RINPriceForecast


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/renewables", tags=["renewables"])

rin_model = RINPriceForecast()
biodiesel_model = BiodieselSpreadAnalysis()
ci_calculator = CarbonIntensityCalculator()

dal = DataAccessLayer(default_price_type="spot")
persistence = RenewablesPersistence()

try:  # Prometheus metrics optional
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - optional metrics path
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

if Counter and Histogram:
    _API_CALL_COUNTER = Counter(
        "ml_service_renewables_api_requests_total",
        "Total renewables API requests served",
        labelnames=("endpoint",),
    )
    _API_LATENCY = Histogram(
        "ml_service_renewables_api_latency_seconds",
        "Latency per renewables API endpoint",
        labelnames=("endpoint",),
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )
else:  # pragma: no cover - metrics disabled
    _API_CALL_COUNTER = None
    _API_LATENCY = None


def _instrument(endpoint: str):
    def decorator(func):
        if Counter is None or Histogram is None:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            if _API_CALL_COUNTER is not None:
                _API_CALL_COUNTER.labels(endpoint=endpoint).inc()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                if _API_LATENCY is not None:
                    duration = time.perf_counter() - start
                    _API_LATENCY.labels(endpoint=endpoint).observe(duration)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RINForecastRequest(BaseModel):
    as_of_date: date
    categories: List[str] = Field(default_factory=lambda: ["D4", "D5", "D6"])
    horizon_days: int = Field(90, ge=7, le=365)
    model_version: str = "v1"


class RINForecastResponse(BaseModel):
    as_of_date: date
    forecasts: List[Dict[str, Any]]
    persisted_rows: int


class BiodieselSpreadRequest(BaseModel):
    as_of_date: date
    region: Optional[str] = None
    lookback_days: int = Field(365, ge=60, le=1825)
    biodiesel_instrument: str = "BIO.BIODIESEL"
    diesel_instrument: str = "OIL.ULSD"
    rin_category: str = "D4"
    btc_value: float = 1.0
    lcfs_value: float = 0.5
    model_version: str = "v1"


class BiodieselSpreadResponse(BaseModel):
    as_of_date: date
    region: Optional[str]
    summary: Dict[str, Any]
    persisted_rows: int


class CarbonIntensityRequest(BaseModel):
    as_of_date: date
    fuel_type: str
    pathway: str = "conventional"
    include_land_use: bool = True
    transport_distance: float = Field(1000.0, ge=0)
    transport_mode: str = Field("truck", regex="^(truck|rail|ship)$")
    model_version: str = "v1"


class CarbonIntensityResponse(BaseModel):
    as_of_date: date
    fuel_type: str
    pathway: str
    total_ci: float
    breakdown: Dict[str, Any]
    persisted_rows: int


class PolicyScenario(BaseModel):
    name: str
    description: Optional[str] = None
    rin_category: Optional[str] = "all"
    demand_impact: float = 0.0
    supply_impact: float = 0.0
    incentive_changes: Optional[Dict[str, float]] = None


class PolicyImpactRequest(BaseModel):
    as_of_date: date
    scenarios: List[PolicyScenario]
    rin_categories: List[str] = Field(default_factory=lambda: ["D4", "D5", "D6"])
    biodiesel_instrument: str = "BIO.BIODIESEL"
    diesel_instrument: str = "OIL.ULSD"
    model_version: str = "v1"


class PolicyImpactResponse(BaseModel):
    as_of_date: date
    results: Dict[str, Any]
    persisted_rows: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_series(
    instrument_id: str,
    *,
    as_of: date,
    lookback_days: int,
) -> pd.Series:
    series = dal.get_price_series(
        instrument_id,
        end=as_of,
        lookback_days=lookback_days,
        price_type="spot",
    )
    if series.empty:
        raise HTTPException(status_code=400, detail=f"No price data for {instrument_id}")
    filtered = series[series.index <= pd.Timestamp(as_of)]
    if filtered.empty:
        raise HTTPException(status_code=400, detail=f"No price data for {instrument_id} before {as_of}")
    return filtered.sort_index()


def _latest_price(instrument_id: str, as_of: date) -> float:
    series = dal.get_price_series(
        instrument_id,
        end=as_of,
        lookback_days=120,
        price_type="spot",
    )
    if series.empty:
        raise HTTPException(status_code=400, detail=f"No price data for {instrument_id}")
    data = series[series.index <= pd.Timestamp(as_of)]
    if data.empty:
        raise HTTPException(status_code=400, detail=f"No price observations for {instrument_id} before {as_of}")
    return float(data.iloc[-1])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/rin-forecast", response_model=RINForecastResponse)
@_instrument("rin_forecast")
def run_rin_forecast(request: RINForecastRequest) -> RINForecastResponse:
    historical_prices: Dict[str, pd.Series] = {}
    for category in request.categories:
        instrument = f"RIN.{category.upper()}"
        series = _price_series(instrument, as_of=request.as_of_date, lookback_days=365)
        historical_prices[category.upper()] = series

    if not historical_prices:
        raise HTTPException(status_code=400, detail="No historical RIN prices available for requested categories")

    forecasts = rin_model.forecast_rin_prices(
        historical_prices,
        forecast_horizon=request.horizon_days,
    )

    records: List[Dict[str, Any]] = []
    response_payload: List[Dict[str, Any]] = []

    for category, series in forecasts.items():
        if series.empty:
            continue
        base_date = pd.to_datetime(request.as_of_date)
        volatility = float(historical_prices[category].diff().std()) if category in historical_prices else 0.0
        for forecast_date, price in series.items():
            horizon = (forecast_date.date() - base_date.date()).days
            records.append(
                {
                    "as_of_date": request.as_of_date,
                    "rin_category": category,
                    "horizon_days": horizon,
                    "forecast_date": forecast_date.date(),
                    "forecast_price": float(price),
                    "std": volatility,
                    "drivers": {"method": "regression", "volatility": volatility},
                    "model_version": request.model_version,
                }
            )
        response_payload.append(
            {
                "category": category,
                "forecast_start": series.index.min().date(),
                "forecast_end": series.index.max().date(),
                "average_price": float(series.mean()),
                "volatility_proxy": volatility,
            }
        )

    if not records:
        raise HTTPException(status_code=500, detail="RIN forecast model returned no results")

    persisted = persistence.persist_rin_forecast(records, default_model_version=request.model_version)

    return RINForecastResponse(
        as_of_date=request.as_of_date,
        forecasts=response_payload,
        persisted_rows=persisted,
    )


@router.post("/biodiesel-spread", response_model=BiodieselSpreadResponse)
@_instrument("biodiesel_spread")
def run_biodiesel_spread(request: BiodieselSpreadRequest) -> BiodieselSpreadResponse:
    biodiesel_prices = _price_series(
        request.biodiesel_instrument,
        as_of=request.as_of_date,
        lookback_days=request.lookback_days,
    )
    diesel_prices = _price_series(
        request.diesel_instrument,
        as_of=request.as_of_date,
        lookback_days=request.lookback_days,
    )

    rin_category = request.rin_category.upper()
    rin_price = _latest_price(f"RIN.{rin_category}", request.as_of_date)

    spread = biodiesel_model.analyze_biodiesel_diesel_spread(
        biodiesel_prices,
        diesel_prices,
        rin_value=rin_price,
        btc_value=request.btc_value + request.lcfs_value,
    )
    if "error" in spread:
        raise HTTPException(status_code=400, detail=spread["error"])

    record = {
        "as_of_date": request.as_of_date,
        "region": request.region,
        "mean_gross_spread": spread["spread_statistics"]["mean_gross_spread"],
        "mean_net_spread": spread["spread_statistics"]["mean_net_spread"],
        "spread_volatility": spread["spread_statistics"].get("spread_volatility"),
        "arbitrage_opportunities": spread.get("arbitrage_opportunities", 0),
        "diagnostics": spread,
        "model_version": request.model_version,
    }

    persisted = persistence.persist_biodiesel_spread([record], default_model_version=request.model_version)

    return BiodieselSpreadResponse(
        as_of_date=request.as_of_date,
        region=request.region,
        summary=spread,
        persisted_rows=persisted,
    )


@router.post("/carbon-intensity", response_model=CarbonIntensityResponse)
@_instrument("carbon_intensity")
def run_carbon_intensity(request: CarbonIntensityRequest) -> CarbonIntensityResponse:
    result = ci_calculator.calculate_fuel_carbon_intensity(
        request.fuel_type,
        pathway=request.pathway,
        include_land_use=request.include_land_use,
        transport_distance=request.transport_distance,
        transport_mode=request.transport_mode,
    )

    record = {
        "as_of_date": request.as_of_date,
        "fuel_type": request.fuel_type,
        "pathway": request.pathway,
        "total_ci": result.get("total_carbon_intensity") or result.get("ci_per_mj"),
        "base_emissions": result.get("base_emissions"),
        "transport_emissions": result.get("transport_emissions"),
        "land_use_emissions": result.get("land_use_emissions"),
        "ci_per_mj": result.get("ci_per_mj"),
        "assumptions": {
            "include_land_use": request.include_land_use,
            "transport_distance": request.transport_distance,
            "transport_mode": request.transport_mode,
        },
        "model_version": request.model_version,
    }

    persisted = persistence.persist_carbon_intensity([record], default_model_version=request.model_version)

    return CarbonIntensityResponse(
        as_of_date=request.as_of_date,
        fuel_type=request.fuel_type,
        pathway=request.pathway,
        total_ci=record["total_ci"],
        breakdown={
            "base_emissions": record["base_emissions"],
            "transport_emissions": record["transport_emissions"],
            "land_use_emissions": record["land_use_emissions"],
        },
        persisted_rows=persisted,
    )


@router.post("/policy-impact", response_model=PolicyImpactResponse)
@_instrument("policy_impact")
def run_policy_impact(request: PolicyImpactRequest) -> PolicyImpactResponse:
    if not request.scenarios:
        raise HTTPException(status_code=400, detail="At least one policy scenario is required")

    current_rin_prices = {}
    for category in request.rin_categories:
        instrument = f"RIN.{category.upper()}"
        current_rin_prices[category.upper()] = _latest_price(instrument, request.as_of_date)

    rin_policy = rin_model.model_policy_impact(
        current_rin_prices,
        [scenario.dict() for scenario in request.scenarios],
    )

    biodiesel_price = _latest_price(request.biodiesel_instrument, request.as_of_date)
    diesel_price = _latest_price(request.diesel_instrument, request.as_of_date)
    biodiesel_policy = biodiesel_model.calculate_policy_incentive_impact(
        biodiesel_price,
        diesel_price,
        [scenario.dict() for scenario in request.scenarios],
    )

    records: List[Dict[str, Any]] = []

    for scenario in request.scenarios:
        scenario_name = scenario.name
        rin_impacts = rin_policy["policy_scenarios"].get(scenario_name, {}).get("impacts", {})
        for category, impact in rin_impacts.items():
            records.append(
                {
                    "as_of_date": request.as_of_date,
                    "policy": scenario_name,
                    "entity": category,
                    "metric": "rin_price_change_pct",
                    "value": impact.get("price_change_pct", 0.0),
                    "details": impact,
                    "model_version": request.model_version,
                }
            )

        biodiesel_impacts = biodiesel_policy["policy_scenarios"].get(scenario_name, {})
        if biodiesel_impacts:
            records.append(
                {
                    "as_of_date": request.as_of_date,
                    "policy": scenario_name,
                    "entity": "biodiesel",
                    "metric": "profitability",
                    "value": biodiesel_impacts.get("profitability", 0.0),
                    "details": biodiesel_impacts,
                    "model_version": request.model_version,
                }
            )

    if not records:
        raise HTTPException(status_code=500, detail="Policy impact scenarios produced no results")

    persisted = persistence.persist_policy_impact(records, default_model_version=request.model_version)

    results = {
        "rin_policy": rin_policy,
        "biodiesel_policy": biodiesel_policy,
    }

    return PolicyImpactResponse(
        as_of_date=request.as_of_date,
        results=results,
        persisted_rows=persisted,
    )
