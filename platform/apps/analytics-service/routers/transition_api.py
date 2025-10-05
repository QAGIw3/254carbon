"""FastAPI router exposing energy transition analytics endpoints."""

from __future__ import annotations

import logging
import time
from datetime import date
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from carbon_price_forecast import CarbonPriceForecast
from data_access import DataAccessLayer
from decarbonization_pathway_model import DecarbonizationPathwayModel
from transition_persistence import TransitionPersistence


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/transition", tags=["transition"])

transition_model = DecarbonizationPathwayModel()
carbon_model = CarbonPriceForecast()
dal = DataAccessLayer(default_price_type="spot")
persistence = TransitionPersistence()

try:  # Prometheus metrics optional
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - optional metrics path
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

if Counter and Histogram:
    _API_CALL_COUNTER = Counter(
        "ml_service_transition_api_requests_total",
        "Total transition API requests served",
        labelnames=("endpoint",),
    )
    _API_LATENCY = Histogram(
        "ml_service_transition_api_latency_seconds",
        "Latency per transition API endpoint",
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


def _flatten_policy_scenarios(
    scenarios: Optional[List["PolicyScenarioInput"]],
) -> Optional[List[Dict[str, Any]]]:
    if not scenarios:
        return None

    payloads: List[Dict[str, Any]] = []
    for scenario in scenarios:
        data = scenario.model_dump()
        market_factors = data.pop("market_factors", {})
        for market, factor in market_factors.items():
            data[f"{market}_factor"] = factor
        payloads.append(data)
    return payloads


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_payload(v) for v in value]
    if isinstance(value, (np.generic,)):  # type: ignore[attr-defined]
        return float(value)
    if isinstance(value, pd.Series):
        return _normalize_payload(value.to_dict())
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PolicyScenarioInput(BaseModel):
    name: str
    description: Optional[str] = None
    price_impact: float = 0.0
    market_factors: Dict[str, float] = Field(default_factory=dict)


class DecarbonizationPathwayRequest(BaseModel):
    as_of_date: date = Field(default_factory=date.today)
    sector: str
    policy_scenario: str = "ambitious"
    target_year: int = Field(2050, ge=2025, le=2100)
    technology_mix: Optional[Dict[str, float]] = None
    model_version: str = "v1"


class EmissionsTrajectoryPoint(BaseModel):
    year: int
    emissions: float


class DecarbonizationPathwayResponse(BaseModel):
    as_of_date: date
    sector: str
    policy_scenario: str
    target_year: int
    annual_reduction_rate: float
    cumulative_emissions: float
    target_achieved: bool
    emissions_trajectory: List[EmissionsTrajectoryPoint]
    technology_analysis: Dict[str, Any]
    persisted_rows: int


class RenewableAdoptionRequest(BaseModel):
    as_of_date: date = Field(default_factory=date.today)
    technology: str
    current_capacity: float = Field(..., gt=0)
    policy_support: float = Field(1.0, ge=0.0, le=5.0)
    economic_factors: Dict[str, float] = Field(default_factory=dict)
    model_version: str = "v1"


class RenewableAdoptionPoint(BaseModel):
    forecast_year: date
    capacity_gw: float


class RenewableAdoptionResponse(BaseModel):
    as_of_date: date
    technology: str
    policy_support: float
    economic_factors: Dict[str, float]
    points: List[RenewableAdoptionPoint]
    persisted_rows: int


class StrandedAssetRiskRequest(BaseModel):
    as_of_date: date = Field(default_factory=date.today)
    asset_values: Dict[str, float]
    asset_lifetimes: Dict[str, int]
    carbon_markets: List[str] = Field(default_factory=lambda: ["eua", "cca", "rggi"])
    forecast_horizon_days: int = Field(365, ge=30, le=1095)
    lookback_days: int = Field(720, ge=90, le=1825)
    policy_scenarios: Optional[List[PolicyScenarioInput]] = None
    risk_free_rate: float = Field(0.05, ge=0.0, le=0.2)
    model_version: str = "v1"


class StrandedAssetItem(BaseModel):
    asset_type: str
    asset_value: float
    carbon_cost_pv: float
    stranded_value: float
    stranded_ratio: float
    remaining_lifetime: int
    risk_level: str
    details: Dict[str, Any]


class StrandedAssetRiskResponse(BaseModel):
    as_of_date: date
    portfolio_stranded_ratio: float
    total_stranded_value: float
    total_asset_value: float
    items: List[StrandedAssetItem]
    persisted_rows: int


# ---------------------------------------------------------------------------
# Endpoint implementations (callable for jobs/tests)
# ---------------------------------------------------------------------------


def run_decarbonization_pathway(
    request: DecarbonizationPathwayRequest,
) -> DecarbonizationPathwayResponse:
    logger.info(
        "Running decarbonization pathway for sector=%s scenario=%s",
        request.sector,
        request.policy_scenario,
    )

    result = transition_model.model_sector_decarbonization(
        request.sector,
        target_year=request.target_year,
        policy_scenario=request.policy_scenario,
        technology_mix=request.technology_mix,
    )

    trajectory_series: pd.Series = result["emissions_trajectory"]
    emissions_points = [
        EmissionsTrajectoryPoint(year=int(year), emissions=float(value))
        for year, value in trajectory_series.items()
    ]

    technology_analysis = _normalize_payload(result.get("technology_analysis", {}))

    persisted_rows = persistence.persist_decarbonization_pathways([
        {
            "as_of_date": request.as_of_date,
            "sector": request.sector,
            "policy_scenario": request.policy_scenario,
            "target_year": request.target_year,
            "annual_reduction_rate": result.get("annual_reduction_rate"),
            "cumulative_emissions": result.get("cumulative_emissions"),
            "target_achieved": result.get("target_achieved", False),
            "emissions_trajectory": trajectory_series.to_dict(),
            "technology_analysis": technology_analysis,
            "model_version": request.model_version,
        }
    ])

    return DecarbonizationPathwayResponse(
        as_of_date=request.as_of_date,
        sector=request.sector,
        policy_scenario=request.policy_scenario,
        target_year=request.target_year,
        annual_reduction_rate=float(result.get("annual_reduction_rate", 0.0)),
        cumulative_emissions=float(result.get("cumulative_emissions", 0.0)),
        target_achieved=bool(result.get("target_achieved", False)),
        emissions_trajectory=emissions_points,
        technology_analysis=technology_analysis,
        persisted_rows=persisted_rows,
    )


def run_renewable_adoption(
    request: RenewableAdoptionRequest,
) -> RenewableAdoptionResponse:
    economic_factors = request.economic_factors or {
        "cost_competitiveness": 1.0,
        "grid_integration": 1.0,
        "financing_availability": 1.0,
    }

    logger.info(
        "Running renewable adoption forecast for technology=%s", request.technology
    )

    adoption_series = transition_model.forecast_renewable_adoption_curves(
        request.technology,
        request.current_capacity,
        policy_support=request.policy_support,
        economic_factors=economic_factors,
    )

    points = [
        RenewableAdoptionPoint(
            forecast_year=ts.date(),
            capacity_gw=float(value),
        )
        for ts, value in adoption_series.items()
    ]

    economic_multiplier = float(np.prod(list(economic_factors.values()))) if economic_factors else 1.0

    persisted_rows = persistence.persist_renewable_adoption_forecast([
        {
            "as_of_date": request.as_of_date,
            "technology": request.technology,
            "forecast_year": point.forecast_year,
            "capacity_gw": point.capacity_gw,
            "policy_support": request.policy_support,
            "economic_multipliers": {
                "factors": economic_factors,
                "composite_multiplier": economic_multiplier,
            },
            "assumptions": {
                "current_capacity": request.current_capacity,
                "max_capacity_multiple": 10,
            },
            "model_version": request.model_version,
        }
        for point in points
    ])

    return RenewableAdoptionResponse(
        as_of_date=request.as_of_date,
        technology=request.technology,
        policy_support=request.policy_support,
        economic_factors=economic_factors,
        points=points,
        persisted_rows=persisted_rows,
    )


def run_stranded_asset_risk(
    request: StrandedAssetRiskRequest,
) -> StrandedAssetRiskResponse:
    if not request.asset_values:
        raise HTTPException(status_code=400, detail="asset_values payload is required")
    if not request.asset_lifetimes:
        raise HTTPException(status_code=400, detail="asset_lifetimes payload is required")

    scenario_payloads = _flatten_policy_scenarios(request.policy_scenarios)

    historical_prices: Dict[str, pd.Series] = {}
    for market in request.carbon_markets:
        series = dal.get_carbon_price_history(
            market,
            end=request.as_of_date,
            lookback_days=request.lookback_days,
        )
        if series.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No carbon price history available for market '{market}'",
            )
        historical_prices[market] = series.sort_index()

    logger.info(
        "Forecasting carbon prices for markets=%s horizon=%s days",
        request.carbon_markets,
        request.forecast_horizon_days,
    )

    price_forecasts = carbon_model.forecast_carbon_prices(
        historical_prices,
        policy_scenarios=scenario_payloads,
        forecast_horizon=request.forecast_horizon_days,
    )

    risk_result = transition_model.analyze_stranded_asset_risk(
        request.asset_values,
        carbon_prices=price_forecasts,
        asset_lifetimes=request.asset_lifetimes,
        risk_free_rate=request.risk_free_rate,
    )

    infrastructure = dal.get_infrastructure_assets(request.asset_values.keys())
    infra_lookup = (
        infrastructure.set_index("asset_type").to_dict(orient="index")
        if not infrastructure.empty
        else {}
    )

    scenario_names = (
        [scenario.name for scenario in request.policy_scenarios]
        if request.policy_scenarios
        else []
    )

    items: List[StrandedAssetItem] = []
    persistence_payload: List[Dict[str, Any]] = []
    for asset_type, metrics in risk_result.get("stranded_risks", {}).items():
        details = {
            "markets": request.carbon_markets,
            "policy_scenarios": scenario_names,
            "risk_free_rate": request.risk_free_rate,
            "infrastructure": infra_lookup.get(asset_type.lower()),
        }
        items.append(
            StrandedAssetItem(
                asset_type=asset_type,
                asset_value=float(metrics.get("asset_value", 0.0)),
                carbon_cost_pv=float(metrics.get("carbon_cost_pv", 0.0)),
                stranded_value=float(metrics.get("stranded_value", 0.0)),
                stranded_ratio=float(metrics.get("stranded_ratio", 0.0)),
                remaining_lifetime=int(metrics.get("remaining_lifetime", 0)),
                risk_level=str(metrics.get("risk_level", "unknown")),
                details=_normalize_payload(details),
            )
        )
        persistence_payload.append(
            {
                "as_of_date": request.as_of_date,
                "asset_type": asset_type,
                "asset_value": metrics.get("asset_value"),
                "carbon_cost_pv": metrics.get("carbon_cost_pv"),
                "stranded_value": metrics.get("stranded_value"),
                "stranded_ratio": metrics.get("stranded_ratio"),
                "remaining_lifetime": metrics.get("remaining_lifetime"),
                "risk_level": metrics.get("risk_level"),
                "details": details,
                "model_version": request.model_version,
            }
        )

    persisted_rows = persistence.persist_stranded_asset_risk(persistence_payload)

    return StrandedAssetRiskResponse(
        as_of_date=request.as_of_date,
        portfolio_stranded_ratio=float(risk_result.get("portfolio_stranded_ratio", 0.0)),
        total_stranded_value=float(risk_result.get("total_stranded_value", 0.0)),
        total_asset_value=float(risk_result.get("total_asset_value", 0.0)),
        items=items,
        persisted_rows=persisted_rows,
    )


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------


@router.post("/decarbonization-pathway", response_model=DecarbonizationPathwayResponse)
@_instrument("decarbonization_pathway")
def decarbonization_pathway_endpoint(
    request: DecarbonizationPathwayRequest,
) -> DecarbonizationPathwayResponse:
    return run_decarbonization_pathway(request)


@router.post("/renewable-adoption", response_model=RenewableAdoptionResponse)
@_instrument("renewable_adoption")
def renewable_adoption_endpoint(
    request: RenewableAdoptionRequest,
) -> RenewableAdoptionResponse:
    return run_renewable_adoption(request)


@router.post("/stranded-asset-risk", response_model=StrandedAssetRiskResponse)
@_instrument("stranded_asset_risk")
def stranded_asset_risk_endpoint(
    request: StrandedAssetRiskRequest,
) -> StrandedAssetRiskResponse:
    return run_stranded_asset_risk(request)


__all__ = [
    "router",
    "run_decarbonization_pathway",
    "run_renewable_adoption",
    "run_stranded_asset_risk",
    "PolicyScenarioInput",
    "DecarbonizationPathwayRequest",
    "RenewableAdoptionRequest",
    "StrandedAssetRiskRequest",
]

