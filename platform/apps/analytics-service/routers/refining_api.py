"""FastAPI router exposing refining analytics endpoints."""

from __future__ import annotations

import logging
import time
from datetime import date
from functools import wraps
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from crack_spread_optimization import CrackSpreadOptimizer
from data_access import DataAccessLayer
from product_demand_elasticity import ProductDemandElasticity
from refinery_yield_model import RefineryYieldModel
from refining_persistence import RefiningPersistence
from transportation_fuel_substitution import TransportationFuelSubstitution


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/refining", tags=["refining"])

optimizer = CrackSpreadOptimizer()
yield_model = RefineryYieldModel()
elasticity_model = ProductDemandElasticity()
substitution_model = TransportationFuelSubstitution()

dal = DataAccessLayer(default_price_type="spot")
persistence = RefiningPersistence()

try:  # Prometheus metrics optional
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - metrics optionality
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

if Counter and Histogram:
    _API_CALL_COUNTER = Counter(
        "ml_service_refining_api_requests_total",
        "Total refining API requests served",
        labelnames=("endpoint",),
    )
    _API_LATENCY = Histogram(
        "ml_service_refining_api_latency_seconds",
        "Latency per refining API endpoint",
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


class CrackOptimizeRequest(BaseModel):
    as_of_date: date
    region: str
    crack_types: List[str] = Field(default_factory=lambda: ["3:2:1", "5:3:2"])
    crude_code: str = Field("OIL.WTI", description="Crude instrument identifier")
    refinery_id: Optional[str] = None
    refinery_constraints: Optional[Dict[str, float]] = None
    price_overrides: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional overrides for product prices (USD/gal for products, USD/bbl for crude)",
    )
    lookback_days: int = Field(120, ge=30, le=720)
    model_version: str = Field("v1", min_length=1)

    @validator("crack_types", pre=True, always=True)
    def _ensure_crack_types(cls, value: Any) -> List[str]:  # noqa: N805
        if not value:
            raise ValueError("At least one crack spread type must be provided")
        return value


class CrackOptimizeResponse(BaseModel):
    as_of_date: date
    region: str
    crack_results: List[Dict[str, Any]]
    persisted_rows: int


class RefineryYieldRequest(BaseModel):
    as_of_date: date
    crude_type: str = Field("wti", min_length=2)
    region: str
    process_config: Optional[Dict[str, float]] = None
    operating_constraints: Optional[Dict[str, float]] = None
    product_price_overrides: Optional[Dict[str, float]] = None
    model_version: str = "v1"


class RefineryYieldResponse(BaseModel):
    as_of_date: date
    crude_type: str
    region: str
    yields: Dict[str, float]
    value_per_bbl: float
    net_value: float
    persisted_rows: int


class ElasticityRequest(BaseModel):
    as_of_date: date
    product: str = Field("gasoline", min_length=3)
    region: Optional[str] = "US"
    method: str = Field("regression", description="Estimation method")
    price_instrument_id: str = Field("OIL.RBOB", min_length=3)
    demand_entity_id: str = Field("US", min_length=2)
    demand_variable: str = Field("gasoline_demand", min_length=3)
    cross_product: Optional[str] = None
    cross_price_instrument_id: Optional[str] = None
    lookback_days: int = Field(365, ge=90, le=1825)
    model_version: str = "v1"


class ElasticityResponse(BaseModel):
    as_of_date: date
    product: str
    region: Optional[str]
    method: str
    results: List[Dict[str, Any]]
    persisted_rows: int


class FuelSubstitutionRequest(BaseModel):
    as_of_date: date
    region: str
    demand_entity_id: str = Field("US", min_length=2)
    gasoline_demand_variable: str = "gasoline_demand"
    diesel_demand_variable: str = "diesel_demand"
    gasoline_price_instrument: str = "OIL.RBOB"
    diesel_price_instrument: str = "OIL.ULSD"
    ev_adoption_rate: float = Field(0.08, ge=0, le=1)
    forecast_years: int = Field(10, ge=1, le=25)
    infrastructure_capacity: Optional[Dict[str, float]] = None
    growth_rates: Optional[Dict[str, float]] = None
    lookback_days: int = Field(365, ge=90, le=1825)
    model_version: str = "v1"


class FuelSubstitutionResponse(BaseModel):
    as_of_date: date
    region: str
    metrics: List[Dict[str, Any]]
    persisted_rows: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_price(
    instrument_id: Optional[str],
    *,
    as_of: date,
    lookback_days: int,
) -> float:
    if not instrument_id:
        raise HTTPException(status_code=400, detail="Instrument identifier is required")

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
        raise HTTPException(status_code=400, detail=f"No price observations for {instrument_id} on or before {as_of}")

    return float(filtered.iloc[-1])


def _load_demand_series(
    entity_id: str,
    variable: str,
    *,
    as_of: date,
    lookback_days: int,
) -> pd.Series:
    series = dal.get_fundamental_series(
        entity_id,
        variable,
        end=as_of,
        lookback_days=lookback_days,
    )
    if series.empty:
        raise HTTPException(status_code=400, detail=f"No fundamentals for {entity_id}/{variable}")
    return series.sort_index()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/crack-optimize", response_model=CrackOptimizeResponse)
@_instrument("crack_optimize")
def run_crack_optimization(request: CrackOptimizeRequest) -> CrackOptimizeResponse:
    gasoline_price = request.price_overrides.get("gasoline") if request.price_overrides else None
    diesel_price = request.price_overrides.get("diesel") if request.price_overrides else None
    jet_price = request.price_overrides.get("jet") if request.price_overrides else None
    crude_price = request.price_overrides.get("crude") if request.price_overrides else None

    if gasoline_price is None:
        gasoline_price = _latest_price("OIL.RBOB", as_of=request.as_of_date, lookback_days=request.lookback_days)
    if diesel_price is None:
        diesel_price = _latest_price("OIL.ULSD", as_of=request.as_of_date, lookback_days=request.lookback_days)
    if jet_price is None:
        try:
            jet_price = _latest_price("OIL.JET", as_of=request.as_of_date, lookback_days=request.lookback_days)
        except HTTPException:
            jet_price = None
    if crude_price is None:
        crude_price = _latest_price(request.crude_code, as_of=request.as_of_date, lookback_days=request.lookback_days)

    product_prices = {
        "gasoline": gasoline_price,
        "diesel": diesel_price,
    }
    if jet_price is not None:
        product_prices["jet_fuel"] = jet_price

    crude_prices = {request.crude_code: crude_price}

    constraints = request.refinery_constraints or {
        "min_gasoline": 0.35,
        "max_gasoline": 0.55,
        "min_diesel": 0.20,
        "max_diesel": 0.45,
        "jet_fuel_ratio": 0.15,
    }

    slate_result = optimizer.optimize_product_slate(
        crude_prices,
        product_prices,
        refinery_constraints=constraints,
    )

    crack_results: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for crack_type in request.crack_types:
        try:
            crack_analysis = optimizer.calculate_crack_spread(
                crude_price,
                gasoline_price,
                diesel_price,
                jet_price,
                crack_type=crack_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        record = {
            "as_of_date": request.as_of_date,
            "region": request.region,
            "refinery_id": request.refinery_id,
            "crack_type": crack_type,
            "crude_code": request.crude_code,
            "gasoline_price": gasoline_price,
            "diesel_price": diesel_price,
            "jet_price": jet_price,
            "crack_spread": crack_analysis.get("crack_spread"),
            "margin_per_bbl": slate_result.get("margin_per_bbl"),
            "optimal_yields": slate_result.get("optimal_product_yields"),
            "constraints": constraints,
            "diagnostics": {
                "crack_analysis": crack_analysis,
                "crude_analysis": slate_result.get("crack_spread_analysis"),
            },
            "model_version": request.model_version,
        }
        records.append(record)

        crack_results.append(
            {
                "crack_type": crack_type,
                "crack_spread": crack_analysis.get("crack_spread"),
                "crack_per_bbl": crack_analysis.get("crack_per_bbl"),
                "product_value": crack_analysis.get("product_value"),
                "margin_per_bbl": slate_result.get("margin_per_bbl"),
                "optimal_yields": slate_result.get("optimal_product_yields"),
                "gasoline_bbl": crack_analysis.get("gasoline_bbl"),
                "diesel_bbl": crack_analysis.get("diesel_bbl"),
                "jet_bbl": crack_analysis.get("jet_bbl"),
            }
        )

    persisted = persistence.persist_crack_optimization(records, default_model_version=request.model_version)

    return CrackOptimizeResponse(
        as_of_date=request.as_of_date,
        region=request.region,
        crack_results=crack_results,
        persisted_rows=persisted,
    )


@router.post("/refinery-yields", response_model=RefineryYieldResponse)
@_instrument("refinery_yields")
def run_refinery_yield(request: RefineryYieldRequest) -> RefineryYieldResponse:
    product_prices = request.product_price_overrides or {}
    for product, instrument in (
        ("gasoline", "OIL.RBOB"),
        ("diesel", "OIL.ULSD"),
        ("fuel_oil", "OIL.HEATOIL"),
    ):
        if product not in product_prices:
            try:
                product_prices[product] = _latest_price(instrument, as_of=request.as_of_date, lookback_days=365)
            except HTTPException:
                continue

    expected_yields = yield_model.predict_yields_from_assay(
        request.crude_type,
        request.process_config or {},
    )

    optimization = yield_model.optimize_process_configuration(
        request.crude_type,
        product_prices,
        operating_constraints=request.operating_constraints,
    )

    record = {
        "as_of_date": request.as_of_date,
        "crude_type": request.crude_type,
        "process_config": optimization.get("optimal_configuration"),
        "yields": optimization.get("expected_yields") or expected_yields,
        "value_per_bbl": optimization.get("value_per_bbl"),
        "operating_cost": optimization.get("operating_cost"),
        "net_value": optimization.get("net_value"),
        "diagnostics": {
            "total_product_value": optimization.get("total_product_value"),
            "constraints": optimization.get("constraints_applied"),
        },
        "model_version": request.model_version,
    }

    persisted = persistence.persist_refinery_yields([record], default_model_version=request.model_version)

    return RefineryYieldResponse(
        as_of_date=request.as_of_date,
        crude_type=request.crude_type,
        region=request.region,
        yields=record["yields"],
        value_per_bbl=record["value_per_bbl"],
        net_value=record["net_value"],
        persisted_rows=persisted,
    )


@router.post("/demand-elasticity", response_model=ElasticityResponse)
@_instrument("demand_elasticity")
def run_product_elasticity(request: ElasticityRequest) -> ElasticityResponse:
    price_series = dal.get_price_series(
        request.price_instrument_id,
        end=request.as_of_date,
        lookback_days=request.lookback_days,
        price_type="spot",
    )
    if price_series.empty:
        raise HTTPException(status_code=400, detail="Price series is empty for elasticity analysis")

    demand_series = _load_demand_series(
        request.demand_entity_id,
        request.demand_variable,
        as_of=request.as_of_date,
        lookback_days=request.lookback_days,
    )

    aligned_prices = price_series[price_series.index.isin(demand_series.index)]
    aligned_demand = demand_series[demand_series.index.isin(aligned_prices.index)]

    if len(aligned_prices) < 20:
        raise HTTPException(status_code=400, detail="Insufficient overlapping data for elasticity estimation")

    result = elasticity_model.estimate_price_elasticity(
        aligned_demand,
        aligned_prices,
        request.product,
        method=request.method,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    records = [
        {
            "as_of_date": request.as_of_date,
            "product": request.product,
            "region": request.region,
            "method": request.method,
            "elasticity": result.get("elasticity"),
            "r_squared": result.get("r_squared"),
            "own_or_cross": "own",
            "product_pair": None,
            "data_points": result.get("data_points", len(aligned_prices)),
            "diagnostics": result,
            "model_version": request.model_version,
        }
    ]

    results_payload = [result]

    if request.cross_product and request.cross_price_instrument_id:
        cross_price_series = dal.get_price_series(
            request.cross_price_instrument_id,
            end=request.as_of_date,
            lookback_days=request.lookback_days,
            price_type="spot",
        )
        cross_demand_series = _load_demand_series(
            request.demand_entity_id,
            request.cross_product,
            as_of=request.as_of_date,
            lookback_days=request.lookback_days,
        )

        demand_data = {
            request.product: aligned_demand,
            request.cross_product: cross_demand_series[cross_demand_series.index.isin(aligned_demand.index)],
        }
        price_data = {
            request.product: aligned_prices,
            request.cross_product: cross_price_series[cross_price_series.index.isin(aligned_prices.index)],
        }

        cross_result = elasticity_model.estimate_cross_price_elasticity(
            demand_data,
            price_data,
            request.product,
            request.cross_product,
        )
        if "error" not in cross_result:
            records.append(
                {
                    "as_of_date": request.as_of_date,
                    "product": request.product,
                    "region": request.region,
                    "method": request.method,
                    "elasticity": cross_result.get("cross_price_elasticity"),
                    "r_squared": cross_result.get("r_squared"),
                    "own_or_cross": "cross",
                    "product_pair": request.cross_product,
                    "data_points": cross_result.get("data_points", len(aligned_prices)),
                    "diagnostics": cross_result,
                    "model_version": request.model_version,
                }
            )
            results_payload.append(cross_result)

    persisted = persistence.persist_product_elasticity(records, default_model_version=request.model_version)

    return ElasticityResponse(
        as_of_date=request.as_of_date,
        product=request.product,
        region=request.region,
        method=request.method,
        results=results_payload,
        persisted_rows=persisted,
    )


@router.post("/fuel-substitution", response_model=FuelSubstitutionResponse)
@_instrument("fuel_substitution")
def run_fuel_substitution(request: FuelSubstitutionRequest) -> FuelSubstitutionResponse:
    gasoline_demand = _load_demand_series(
        request.demand_entity_id,
        request.gasoline_demand_variable,
        as_of=request.as_of_date,
        lookback_days=request.lookback_days,
    )
    diesel_demand = _load_demand_series(
        request.demand_entity_id,
        request.diesel_demand_variable,
        as_of=request.as_of_date,
        lookback_days=request.lookback_days,
    )
    gasoline_prices = dal.get_price_series(
        request.gasoline_price_instrument,
        end=request.as_of_date,
        lookback_days=request.lookback_days,
        price_type="spot",
    )
    diesel_prices = dal.get_price_series(
        request.diesel_price_instrument,
        end=request.as_of_date,
        lookback_days=request.lookback_days,
        price_type="spot",
    )

    substitution = substitution_model.analyze_gasoline_diesel_substitution(
        gasoline_demand,
        diesel_demand,
        gasoline_prices,
        diesel_prices,
    )
    if "error" in substitution:
        raise HTTPException(status_code=400, detail=substitution["error"])

    current_gasoline = float(gasoline_demand.iloc[-1])
    ev_impact = substitution_model.forecast_electric_vehicle_impact(
        current_gasoline,
        request.ev_adoption_rate,
        forecast_years=request.forecast_years,
    )

    infra_capacity = request.infrastructure_capacity or {
        "gasoline": current_gasoline * 1.2,
        "diesel": float(diesel_demand.iloc[-1]) * 1.2,
        "electric": current_gasoline * 0.5,
    }
    growth_rates = request.growth_rates or {"gasoline": -0.01, "diesel": -0.005, "electric": 0.15}

    constraint_analysis = substitution_model.model_infrastructure_constraints(
        {
            "gasoline": current_gasoline,
            "diesel": float(diesel_demand.iloc[-1]),
            "electric": current_gasoline * 0.1,
        },
        infrastructure_capacity=infra_capacity,
        growth_rates=growth_rates,
        constraint_years=request.forecast_years,
    )

    metrics = [
        {
            "metric": "substitution_elasticity",
            "value": substitution.get("substitution_elasticity", 0.0),
            "details": substitution,
            "model_version": request.model_version,
        },
        {
            "metric": "ev_demand_reduction_pct",
            "value": ev_impact.get("demand_reduction_pct", 0.0),
            "details": ev_impact,
            "model_version": request.model_version,
        },
        {
            "metric": "infrastructure_utilization",
            "value": constraint_analysis.get("system_utilization", 0.0),
            "details": constraint_analysis,
            "model_version": request.model_version,
        },
    ]

    records = [
        {
            "as_of_date": request.as_of_date,
            "region": request.region,
            "metric": metric["metric"],
            "value": metric["value"],
            "details": metric["details"],
            "model_version": request.model_version,
        }
        for metric in metrics
    ]

    persisted = persistence.persist_transport_substitution(records, default_model_version=request.model_version)

    return FuelSubstitutionResponse(
        as_of_date=request.as_of_date,
        region=request.region,
        metrics=metrics,
        persisted_rows=persisted,
    )
