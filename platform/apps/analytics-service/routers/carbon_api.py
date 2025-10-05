"""FastAPI router exposing carbon pricing analytics endpoints."""

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
from transition_api import PolicyScenarioInput
from transition_persistence import TransitionPersistence


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/carbon", tags=["carbon"])

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
        "ml_service_carbon_api_requests_total",
        "Total carbon API requests served",
        labelnames=("endpoint",),
    )
    _API_LATENCY = Histogram(
        "ml_service_carbon_api_latency_seconds",
        "Latency per carbon API endpoint",
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
    scenarios: Optional[List[PolicyScenarioInput]],
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


def _load_carbon_history(
    markets: List[str],
    *,
    as_of_date: date,
    lookback_days: int,
) -> Dict[str, pd.Series]:
    history: Dict[str, pd.Series] = {}
    for market in markets:
        series = dal.get_carbon_price_history(
            market,
            end=as_of_date,
            lookback_days=lookback_days,
        )
        if series.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No carbon price history available for market '{market}'",
            )
        history[market] = series.sort_index()
    return history


def _series_from_observations(observations: List["EmissionObservation"]) -> pd.Series:
    if not observations:
        return pd.Series(dtype=float)
    points = sorted(observations, key=lambda obs: obs.period)
    return pd.Series(
        [float(obs.emissions) for obs in points],
        index=[pd.Timestamp(obs.period) for obs in points],
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ForecastPoint(BaseModel):
    forecast_date: date
    forecast_price: float
    std: Optional[float] = None


class MarketForecast(BaseModel):
    market: str
    latest_price: float
    points: List[ForecastPoint]
    diagnostics: Dict[str, Any]


class CarbonPriceForecastRequest(BaseModel):
    as_of_date: date = Field(default_factory=date.today)
    markets: List[str] = Field(default_factory=lambda: ["eua", "cca", "rggi"])
    horizon_days: int = Field(365, ge=30, le=1095)
    lookback_days: int = Field(720, ge=90, le=1825)
    policy_scenarios: Optional[List[PolicyScenarioInput]] = None
    model_version: str = "v1"


class CarbonPriceForecastResponse(BaseModel):
    as_of_date: date
    forecasts: List[MarketForecast]
    persisted_rows: int


class EmissionObservation(BaseModel):
    period: date
    emissions: float


class ComplianceCostRequest(BaseModel):
    as_of_date: date = Field(default_factory=date.today)
    markets: List[str] = Field(default_factory=lambda: ["eua", "cca", "rggi"])
    horizon_days: int = Field(180, ge=30, le=1095)
    lookback_days: int = Field(720, ge=90, le=1825)
    emissions_data: Dict[str, List[EmissionObservation]]
    compliance_obligations: Dict[str, float]
    policy_scenarios: Optional[List[PolicyScenarioInput]] = None
    model_version: str = "v1"


class ComplianceCostResult(BaseModel):
    market: str
    total_emissions: float
    average_price: float
    cost_per_tonne: float
    total_compliance_cost: float
    price_volatility: float
    sector_costs: Dict[str, float]


class ComplianceCostResponse(BaseModel):
    as_of_date: date
    horizon_days: int
    results: List[ComplianceCostResult]
    persisted_rows: int


class CarbonLeakageRiskRequest(BaseModel):
    as_of_date: date = Field(default_factory=date.today)
    domestic_prices: Dict[str, float]
    international_prices: Dict[str, float]
    trade_exposure: Dict[str, float]
    emissions_intensity: Dict[str, float]
    model_version: str = "v1"


class CarbonLeakageItem(BaseModel):
    sector: str
    domestic_price: float
    international_price: float
    price_differential: float
    trade_exposure: float
    emissions_intensity: float
    leakage_risk_score: float
    risk_level: str


class CarbonLeakageResponse(BaseModel):
    as_of_date: date
    items: List[CarbonLeakageItem]
    persisted_rows: int


class PolicyImpactRequest(BaseModel):
    as_of_date: date = Field(default_factory=date.today)
    baseline_prices: Dict[str, float]
    policy_scenarios: List[PolicyScenarioInput]
    scenario_probabilities: Optional[Dict[str, float]] = None
    model_version: str = "v1"


class ScenarioImpactEntry(BaseModel):
    scenario: str
    entity: str
    metrics: Dict[str, Any]


class PolicyImpactResponse(BaseModel):
    as_of_date: date
    expected_prices: Dict[str, float]
    scenarios: List[ScenarioImpactEntry]
    persisted_rows: int


# ---------------------------------------------------------------------------
# Endpoint implementations (callable for jobs/tests)
# ---------------------------------------------------------------------------


def run_carbon_price_forecast(
    request: CarbonPriceForecastRequest,
) -> CarbonPriceForecastResponse:
    history = _load_carbon_history(
        request.markets,
        as_of_date=request.as_of_date,
        lookback_days=request.lookback_days,
    )

    scenario_payloads = _flatten_policy_scenarios(request.policy_scenarios)
    scenario_names = (
        [scenario.name for scenario in request.policy_scenarios]
        if request.policy_scenarios
        else []
    )

    forecasts = carbon_model.forecast_carbon_prices(
        history,
        policy_scenarios=scenario_payloads,
        forecast_horizon=request.horizon_days,
    )

    records: List[Dict[str, Any]] = []
    market_payload: List[MarketForecast] = []
    for market, forecast_series in forecasts.items():
        if forecast_series.empty:
            continue

        std_value = float(forecast_series.std()) if len(forecast_series) > 1 else None
        points: List[ForecastPoint] = []
        for ts, price in forecast_series.items():
            points.append(
                ForecastPoint(
                    forecast_date=ts.date(),
                    forecast_price=float(price),
                    std=std_value,
                )
            )
            records.append(
                {
                    "as_of_date": request.as_of_date,
                    "market": market,
                    "horizon_days": request.horizon_days,
                    "forecast_date": ts.date(),
                    "forecast_price": float(price),
                    "std": std_value,
                    "drivers": {
                        "policy_scenarios": scenario_names,
                        "lookback_days": request.lookback_days,
                    },
                    "model_version": request.model_version,
                }
            )

        latest_series = history.get(market)
        latest_price = float(latest_series.iloc[-1]) if latest_series is not None and not latest_series.empty else float("nan")
        market_payload.append(
            MarketForecast(
                market=market,
                latest_price=latest_price,
                points=points,
                diagnostics={
                    "std": std_value,
                    "scenario_names": scenario_names,
                    "forecast_horizon_days": request.horizon_days,
                },
            )
        )

    persisted_rows = persistence.persist_carbon_price_forecasts(records)

    return CarbonPriceForecastResponse(
        as_of_date=request.as_of_date,
        forecasts=market_payload,
        persisted_rows=persisted_rows,
    )


def run_compliance_costs(
    request: ComplianceCostRequest,
) -> ComplianceCostResponse:
    history = _load_carbon_history(
        request.markets,
        as_of_date=request.as_of_date,
        lookback_days=request.lookback_days,
    )

    scenario_payloads = _flatten_policy_scenarios(request.policy_scenarios)

    forecasts = carbon_model.forecast_carbon_prices(
        history,
        policy_scenarios=scenario_payloads,
        forecast_horizon=request.horizon_days,
    )

    emissions_series: Dict[str, pd.Series] = {}
    for market, observations in request.emissions_data.items():
        emissions_series[market] = _series_from_observations(observations)

    cost_result = carbon_model.analyze_compliance_cost_impact(
        forecasts,
        emissions_data=emissions_series,
        compliance_obligations=request.compliance_obligations,
    )

    response_items: List[ComplianceCostResult] = []
    records: List[Dict[str, Any]] = []
    for market, metrics in cost_result.get("compliance_costs", {}).items():
        sector_costs = metrics.get("sector_costs", {})
        response_items.append(
            ComplianceCostResult(
                market=market,
                total_emissions=float(metrics.get("total_emissions", 0.0)),
                average_price=float(metrics.get("average_price", 0.0)),
                cost_per_tonne=float(metrics.get("cost_per_tonne", 0.0)),
                total_compliance_cost=float(metrics.get("total_compliance_cost", 0.0)),
                price_volatility=float(metrics.get("price_volatility", 0.0)),
                sector_costs=_normalize_payload(sector_costs),
            )
        )
        records.append(
            {
                "as_of_date": request.as_of_date,
                "market": market,
                "sector": "aggregate",
                "total_emissions": metrics.get("total_emissions"),
                "average_price": metrics.get("average_price"),
                "cost_per_tonne": metrics.get("cost_per_tonne"),
                "total_compliance_cost": metrics.get("total_compliance_cost"),
                "details": {
                    "sector_costs": sector_costs,
                    "obligations": request.compliance_obligations,
                },
                "model_version": request.model_version,
            }
        )

    persisted_rows = persistence.persist_compliance_costs(records)

    return ComplianceCostResponse(
        as_of_date=request.as_of_date,
        horizon_days=request.horizon_days,
        results=response_items,
        persisted_rows=persisted_rows,
    )


def run_carbon_leakage_risk(
    request: CarbonLeakageRiskRequest,
) -> CarbonLeakageResponse:
    leakage = carbon_model.model_carbon_leakage_risk(
        request.domestic_prices,
        international_competitor_prices=request.international_prices,
        trade_exposure=request.trade_exposure,
        emissions_intensity=request.emissions_intensity,
    )

    items: List[CarbonLeakageItem] = []
    records: List[Dict[str, Any]] = []
    for sector, metrics in leakage.get("leakage_risks", {}).items():
        items.append(
            CarbonLeakageItem(
                sector=sector,
                domestic_price=float(metrics.get("domestic_carbon_price", 0.0)),
                international_price=float(metrics.get("international_carbon_price", 0.0)),
                price_differential=float(metrics.get("price_differential", 0.0)),
                trade_exposure=float(metrics.get("trade_exposure", 0.0)),
                emissions_intensity=float(metrics.get("emissions_intensity", 0.0)),
                leakage_risk_score=float(metrics.get("leakage_risk_score", 0.0)),
                risk_level=str(metrics.get("risk_level", "low")),
            )
        )
        records.append(
            {
                "as_of_date": request.as_of_date,
                "sector": sector,
                "domestic_price": metrics.get("domestic_carbon_price"),
                "international_price": metrics.get("international_carbon_price"),
                "price_differential": metrics.get("price_differential"),
                "trade_exposure": metrics.get("trade_exposure"),
                "emissions_intensity": metrics.get("emissions_intensity"),
                "leakage_risk_score": metrics.get("leakage_risk_score"),
                "risk_level": metrics.get("risk_level"),
                "details": {
                    "average_leakage_risk": leakage.get("average_leakage_risk"),
                    "overall_risk_level": leakage.get("overall_risk_level"),
                },
                "model_version": request.model_version,
            }
        )

    persisted_rows = persistence.persist_carbon_leakage_risk(records)

    return CarbonLeakageResponse(
        as_of_date=request.as_of_date,
        items=items,
        persisted_rows=persisted_rows,
    )


def run_policy_impact(
    request: PolicyImpactRequest,
) -> PolicyImpactResponse:
    scenario_payloads = _flatten_policy_scenarios(request.policy_scenarios)

    impact = carbon_model.forecast_policy_scenario_impact(
        request.baseline_prices,
        policy_scenarios=scenario_payloads or [],
        scenario_probabilities=request.scenario_probabilities,
    )

    records: List[Dict[str, Any]] = []
    scenario_entries: List[ScenarioImpactEntry] = []

    for scenario_name, payload in impact.get("policy_scenarios", {}).items():
        impacts_by_market = payload.get("impacts", {})
        scenario_entries.append(
            ScenarioImpactEntry(
                scenario=scenario_name,
                entity="aggregate",
                metrics=_normalize_payload(payload),
            )
        )
        for market, metrics in impacts_by_market.items():
            records.extend(
                [
                    {
                        "as_of_date": request.as_of_date,
                        "scenario": scenario_name,
                        "entity": market,
                        "metric": "new_price",
                        "value": metrics.get("new_price"),
                        "details": metrics,
                        "model_version": request.model_version,
                    },
                    {
                        "as_of_date": request.as_of_date,
                        "scenario": scenario_name,
                        "entity": market,
                        "metric": "price_change_pct",
                        "value": metrics.get("price_change_pct"),
                        "details": {
                            "baseline_price": metrics.get("baseline_price"),
                            "scenario_probability": payload.get("scenario_probability"),
                        },
                        "model_version": request.model_version,
                    },
                ]
            )

    expected_prices = impact.get("expected_prices", {})
    for market, value in expected_prices.items():
        records.append(
            {
                "as_of_date": request.as_of_date,
                "scenario": "expected",
                "entity": market,
                "metric": "expected_price",
                "value": value,
                "details": {
                    "baseline": request.baseline_prices.get(market),
                },
                "model_version": request.model_version,
            }
        )

    persisted_rows = persistence.persist_policy_scenario_impacts(records)

    return PolicyImpactResponse(
        as_of_date=request.as_of_date,
        expected_prices=_normalize_payload(expected_prices),
        scenarios=scenario_entries,
        persisted_rows=persisted_rows,
    )


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------


@router.post("/price-forecast", response_model=CarbonPriceForecastResponse)
@_instrument("price_forecast")
def carbon_price_forecast_endpoint(
    request: CarbonPriceForecastRequest,
) -> CarbonPriceForecastResponse:
    return run_carbon_price_forecast(request)


@router.post("/compliance-costs", response_model=ComplianceCostResponse)
@_instrument("compliance_costs")
def compliance_costs_endpoint(
    request: ComplianceCostRequest,
) -> ComplianceCostResponse:
    return run_compliance_costs(request)


@router.post("/leakage-risk", response_model=CarbonLeakageResponse)
@_instrument("leakage_risk")
def carbon_leakage_endpoint(
    request: CarbonLeakageRiskRequest,
) -> CarbonLeakageResponse:
    return run_carbon_leakage_risk(request)


@router.post("/policy-impact", response_model=PolicyImpactResponse)
@_instrument("policy_impact")
def policy_impact_endpoint(
    request: PolicyImpactRequest,
) -> PolicyImpactResponse:
    return run_policy_impact(request)


__all__ = [
    "router",
    "run_carbon_price_forecast",
    "run_compliance_costs",
    "run_carbon_leakage_risk",
    "run_policy_impact",
    "CarbonPriceForecastRequest",
    "ComplianceCostRequest",
    "CarbonLeakageRiskRequest",
    "PolicyImpactRequest",
]

