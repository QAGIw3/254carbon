"""FastAPI router exposing supply chain analytics endpoints."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from coal_transportation_model import CoalTransportationModel
from data_access import DataAccessLayer
from lng_supply_chain_model import LNGSupplyChainModel
from pipeline_congestion_model import PipelineCongestionModel
from seasonal_demand_forecast import SeasonalDemandForecast
from supply_chain_persistence import SupplyChainPersistence

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["supply-chain"])

lng_model = LNGSupplyChainModel()
coal_model = CoalTransportationModel()
pipeline_model = PipelineCongestionModel()
seasonal_model = SeasonalDemandForecast()

dal = DataAccessLayer()
persistence = SupplyChainPersistence()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LNGOptimizeRoutesRequest(BaseModel):
    as_of_date: date
    export_terminals: List[str]
    import_terminals: List[str]
    cargo_size_bcf: float = Field(3.5, gt=0)
    vessel_speed_knots: float = Field(19.5, gt=0)
    fuel_price_usd_per_tonne: float = Field(600.0, ge=0)
    vessel_type: str = Field("standard", min_length=3)
    model_version: str = Field("v1", min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class LNGRoutingSummary(BaseModel):
    export_terminal: str
    import_terminal: str
    distance_nm: float
    voyage_time_days: float
    fuel_cost_usd: float
    charter_cost_usd: float
    total_cost_usd: float
    cost_per_mmbtu_usd: float
    is_optimal_route: bool


class LNGOptimizeRoutesResponse(BaseModel):
    optimal_route: LNGRoutingSummary
    all_routes: List[LNGRoutingSummary]
    average_cost_per_mmbtu: float
    persisted_rows: int


class LNGArbitrageRequest(BaseModel):
    as_of_date: date
    export_prices: Dict[str, float]
    import_prices: Dict[str, float]
    transport_costs: Dict[str, float]
    liquefaction_cost: float = 0.8
    regasification_cost: float = 0.3
    model_version: str = "v1"


class LNGArbitrageResponse(BaseModel):
    best_opportunity: Dict[str, Any]
    total_arbitrage_profit: float
    average_profit_margin: float
    persisted_rows: int


class CoalRouteCostRequest(BaseModel):
    as_of_month: date
    route: str
    cargo_size_tonnes: float = Field(75_000, gt=0)
    vessel_type: Optional[str] = None
    fuel_price_usd_per_tonne: float = Field(600.0, ge=0)
    include_congestion: bool = True
    carbon_price_usd_per_tonne: float = Field(50.0, ge=0)
    model_version: str = "v1"


class CoalRouteCostResponse(BaseModel):
    freight_rate_usd_per_tonne: float
    total_cost_usd: float
    breakdown: Dict[str, float]
    persisted_rows: int


class CoalMultimodalRequest(BaseModel):
    as_of_month: date
    origin: str
    destination: str
    cargo_size_tonnes: float
    transport_options: Dict[str, Dict[str, float]]
    model_version: str = "v1"


class CoalMultimodalResponse(BaseModel):
    optimal_transport: Dict[str, Any]
    all_options: List[Dict[str, Any]]
    total_transport_cost: float
    persisted_rows: int


class PipelineTrainRequest(BaseModel):
    pipeline_id: str
    market: Optional[str] = None
    segment: Optional[str] = None
    flow_entity_id: str
    flow_variable: str
    weather_entity_id: Optional[str] = None
    weather_variables: Optional[List[str]] = None
    demand_entity_id: Optional[str] = None
    demand_variable: Optional[str] = None
    lookback_days: int = 730
    horizon_days: int = 7
    model_version: str = "v1"


class PipelineTrainResponse(BaseModel):
    best_model: str
    metrics: Dict[str, Dict[str, float]]
    feature_importance: Dict[str, Any]
    training_data_size: int
    persisted_rows: int


class SeasonalForecastRequest(BaseModel):
    region: str
    horizon_days: int = 90
    scenario_id: str = "BASE"
    sector: Optional[str] = None
    historical_entity_id: str
    historical_variable: str
    weather_entity_id: str
    weather_variable: str
    economic_indicators: Optional[Dict[str, str]] = None
    model_version: str = "v1"


class SeasonalForecastResponse(BaseModel):
    records: List[Dict[str, Any]]
    persisted_rows: int


class PeakRiskRequest(BaseModel):
    region: str
    historical_entity_id: str
    historical_variable: str
    scenarios: List[Dict[str, Any]]


class PeakRiskResponse(BaseModel):
    historical_peak: float
    extreme_weather_peak: float
    statistical_peak: float
    return_period_peak: float
    details: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_non_empty_series(series: pd.Series, label: str) -> pd.Series:
    if series.empty:
        raise HTTPException(status_code=400, detail=f"No data returned for {label}")
    return series.sort_index()


def _build_weather_frame(entity_id: Optional[str], variables: Optional[List[str]], lookback_days: int) -> pd.DataFrame:
    if not entity_id or not variables:
        return pd.DataFrame()

    frames = []
    for variable in variables:
        series = dal.get_weather_series(
            entity_id,
            variable,
            lookback_days=lookback_days,
        )
        if series.empty:
            logger.warning("Weather series %s/%s returned no data", entity_id, variable)
            continue
        frames.append(series.rename(variable))

    if not frames:
        return pd.DataFrame()

    weather_df = pd.concat(frames, axis=1).sort_index()
    weather_df = weather_df.fillna(method="ffill")
    return weather_df


def _build_demand_series(entity_id: Optional[str], variable: Optional[str], lookback_days: int) -> pd.Series:
    if not entity_id or not variable:
        return pd.Series(dtype=float)
    demand_series = dal.get_fundamental_series(
        entity_id,
        variable,
        lookback_days=lookback_days,
    )
    return demand_series.sort_index()


def _map_risk_to_score(risk: str) -> float:
    mapping = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
    return mapping.get(risk, 0.0)


# ---------------------------------------------------------------------------
# LNG endpoints
# ---------------------------------------------------------------------------


@router.post("/supplychain/lng/optimize-routes", response_model=LNGOptimizeRoutesResponse)
async def optimize_lng_routes(payload: LNGOptimizeRoutesRequest) -> LNGOptimizeRoutesResponse:
    result = lng_model.optimize_lng_routing(
        payload.export_terminals,
        payload.import_terminals,
        cargo_size=payload.cargo_size_bcf,
        vessel_speed=payload.vessel_speed_knots,
        fuel_price=payload.fuel_price_usd_per_tonne,
    )

    optimal = result["optimal_route"]
    optimal_route = LNGRoutingSummary(
        export_terminal=optimal["export_terminal"],
        import_terminal=optimal["import_terminal"],
        distance_nm=optimal["distance_nm"],
        voyage_time_days=optimal["voyage_time_days"],
        fuel_cost_usd=optimal["fuel_cost"],
        charter_cost_usd=optimal["charter_cost"],
        total_cost_usd=optimal["total_cost"],
        cost_per_mmbtu_usd=optimal["cost_per_mmbtu"],
        is_optimal_route=True,
    )

    all_routes = [
        LNGRoutingSummary(
            export_terminal=route["export_terminal"],
            import_terminal=route["import_terminal"],
            distance_nm=route["distance_nm"],
            voyage_time_days=route["voyage_time_days"],
            fuel_cost_usd=route["fuel_cost"],
            charter_cost_usd=route["charter_cost"],
            total_cost_usd=route["total_cost"],
            cost_per_mmbtu_usd=route["cost_per_mmbtu"],
            is_optimal_route=(route == optimal),
        )
        for route in result["all_routes"]
    ]

    persistence_records: List[Dict[str, Any]] = []
    for route in result["all_routes"]:
        route_id = f"{route['export_terminal']}->{route['import_terminal']}"
        persistence_records.append(
            {
                "route_id": route_id,
                "export_terminal": route["export_terminal"],
                "import_terminal": route["import_terminal"],
                "vessel_type": payload.vessel_type,
                "cargo_size_bcf": payload.cargo_size_bcf,
                "vessel_speed_knots": payload.vessel_speed_knots,
                "fuel_price_usd_per_tonne": payload.fuel_price_usd_per_tonne,
                "distance_nm": route["distance_nm"],
                "voyage_time_days": route["voyage_time_days"],
                "fuel_consumption_tonnes": route["fuel_consumption_tonnes"],
                "fuel_cost_usd": route["fuel_cost"],
                "charter_cost_usd": route["charter_cost"],
                "port_cost_usd": None,
                "total_cost_usd": route["total_cost"],
                "cost_per_mmbtu_usd": route["cost_per_mmbtu"],
                "is_optimal_route": int(route == optimal),
            }
        )

    persisted = persistence.persist_lng_routing(
        as_of=payload.as_of_date,
        routes=persistence_records,
        model_version=payload.model_version,
        assumptions=payload.metadata or {
            "cargo_size_bcf": payload.cargo_size_bcf,
            "vessel_speed_knots": payload.vessel_speed_knots,
        },
        diagnostics={"average_cost_per_mmbtu": result["average_cost_per_mmbtu"]},
    )

    return LNGOptimizeRoutesResponse(
        optimal_route=optimal_route,
        all_routes=all_routes,
        average_cost_per_mmbtu=result["average_cost_per_mmbtu"],
        persisted_rows=persisted,
    )


@router.post("/supplychain/lng/arbitrage", response_model=LNGArbitrageResponse)
async def analyze_lng_arbitrage(payload: LNGArbitrageRequest) -> LNGArbitrageResponse:
    analysis = lng_model.analyze_lng_arbitrage_opportunities(
        payload.export_prices,
        payload.import_prices,
        payload.transport_costs,
        liquefaction_cost=payload.liquefaction_cost,
        regasification_cost=payload.regasification_cost,
    )

    opportunities = analysis.get("all_opportunities", [])

    persistence_records = []
    for opportunity in opportunities:
        route_id = f"{opportunity['export_terminal']}->{opportunity['import_terminal']}"
        persistence_records.append(
            {
                "route_id": route_id,
                "export_terminal": opportunity["export_terminal"],
                "import_terminal": opportunity["import_terminal"],
                "vessel_type": "spot",
                "cargo_size_bcf": 0.0,
                "vessel_speed_knots": 0.0,
                "fuel_price_usd_per_tonne": 0.0,
                "distance_nm": 0.0,
                "voyage_time_days": 0.0,
                "fuel_consumption_tonnes": 0.0,
                "fuel_cost_usd": opportunity["transport_cost"] * 1_000_000,  # scaled placeholder
                "charter_cost_usd": 0.0,
                "port_cost_usd": None,
                "total_cost_usd": opportunity["delivered_cost"] * 1_000_000,
                "cost_per_mmbtu_usd": opportunity["delivered_cost"],
                "is_optimal_route": int(opportunity == analysis.get("best_opportunity")),
            }
        )

    persisted = 0
    if persistence_records:
        persisted = persistence.persist_lng_routing(
            as_of=payload.as_of_date,
            routes=persistence_records,
            model_version=payload.model_version,
            assumptions={
                "liquefaction_cost": payload.liquefaction_cost,
                "regasification_cost": payload.regasification_cost,
            },
            diagnostics={
                "total_arbitrage_profit": analysis.get("total_arbitrage_profit", 0),
                "average_profit_margin": analysis.get("average_profit_margin", 0),
            },
        )

    return LNGArbitrageResponse(
        best_opportunity=analysis.get("best_opportunity", {}),
        total_arbitrage_profit=analysis.get("total_arbitrage_profit", 0.0),
        average_profit_margin=analysis.get("average_profit_margin", 0.0),
        persisted_rows=persisted,
    )


# ---------------------------------------------------------------------------
# Coal transportation endpoints
# ---------------------------------------------------------------------------


@router.post("/supplychain/coal/route-cost", response_model=CoalRouteCostResponse)
async def calculate_coal_route_cost(payload: CoalRouteCostRequest) -> CoalRouteCostResponse:
    freight = coal_model.calculate_sea_freight_rates(
        payload.route,
        payload.cargo_size_tonnes,
        vessel_type=payload.vessel_type,
        fuel_price=payload.fuel_price_usd_per_tonne,
        include_congestion=payload.include_congestion,
    )

    carbon_cost = coal_model.calculate_carbon_transport_costs(
        payload.route,
        payload.cargo_size_tonnes,
        carbon_price=payload.carbon_price_usd_per_tonne,
    )

    total_cost = freight["freight_rate_usd_per_tonne"] * payload.cargo_size_tonnes
    base_cost = freight["base_rate"] * payload.cargo_size_tonnes
    seasonal_factor = freight.get("seasonal_adjustment", 1.0)
    congestion_factor = freight.get("congestion_premium", 1.0)
    seasonal_cost = base_cost * (seasonal_factor - 1)
    congestion_cost = (
        base_cost * seasonal_factor * (congestion_factor - 1)
        if payload.include_congestion and congestion_factor > 1
        else 0.0
    )

    breakdown = {
        "fuel_cost_usd": freight["fuel_cost"],
        "charter_cost_usd": freight["charter_cost"],
        "port_cost_usd": freight["port_cost"],
        "seasonal_adjustment_usd": seasonal_cost,
        "congestion_premium_usd": congestion_cost,
        "carbon_cost_usd": carbon_cost["carbon_cost_usd"],
    }

    persistence_record = {
        "route_id": payload.route,
        "origin_region": payload.route.split("_to_")[0],
        "destination_region": payload.route.split("_to_")[-1],
        "transport_mode": "sea",
        "vessel_type": freight.get("vessel_type", payload.vessel_type),
        "cargo_tonnes": payload.cargo_size_tonnes,
        "fuel_price_usd_per_tonne": payload.fuel_price_usd_per_tonne,
        "freight_cost_usd": freight["freight_rate_usd_per_tonne"] * payload.cargo_size_tonnes,
        "bunker_cost_usd": freight["fuel_cost"],
        "port_fees_usd": freight["port_cost"],
        "congestion_premium_usd": congestion_cost,
        "carbon_cost_usd": carbon_cost["carbon_cost_usd"],
        "demurrage_cost_usd": None,
        "rail_cost_usd": None,
        "truck_cost_usd": None,
        "total_cost_usd": total_cost,
    }

    persisted = persistence.persist_coal_transport_costs(
        as_of_month=payload.as_of_month,
        records=[persistence_record],
        model_version=payload.model_version,
        assumptions={"include_congestion": payload.include_congestion},
        diagnostics={"freight_rate_usd_per_tonne": freight["freight_rate_usd_per_tonne"]},
    )

    return CoalRouteCostResponse(
        freight_rate_usd_per_tonne=freight["freight_rate_usd_per_tonne"],
        total_cost_usd=total_cost,
        breakdown=breakdown,
        persisted_rows=persisted,
    )


@router.post("/supplychain/coal/multimodal", response_model=CoalMultimodalResponse)
async def optimize_coal_multimodal(payload: CoalMultimodalRequest) -> CoalMultimodalResponse:
    optimisation = coal_model.optimize_multi_modal_transport(
        payload.origin,
        payload.destination,
        payload.cargo_size_tonnes,
        payload.transport_options,
    )

    persistence_records = []
    for option in optimisation["all_options"]:
        persistence_records.append(
            {
                "route_id": f"{payload.origin}_to_{payload.destination}",
                "origin_region": payload.origin,
                "destination_region": payload.destination,
                "transport_mode": option["mode"],
                "vessel_type": option.get("mode"),
                "cargo_tonnes": payload.cargo_size_tonnes,
                "fuel_price_usd_per_tonne": 0.0,
                "freight_cost_usd": option["total_cost"],
                "bunker_cost_usd": None,
                "port_fees_usd": None,
                "congestion_premium_usd": None,
                "carbon_cost_usd": None,
                "demurrage_cost_usd": None,
                "rail_cost_usd": option["total_cost"] if option["mode"] == "rail" else None,
                "truck_cost_usd": option["total_cost"] if option["mode"] == "truck" else None,
                "total_cost_usd": option["total_cost"],
            }
        )

    persisted = persistence.persist_coal_transport_costs(
        as_of_month=payload.as_of_month,
        records=persistence_records,
        model_version=payload.model_version,
        assumptions={"transport_options": payload.transport_options},
        diagnostics={"optimal_mode": optimisation["optimal_transport"]["mode"]},
    )

    return CoalMultimodalResponse(
        optimal_transport=optimisation["optimal_transport"],
        all_options=optimisation["all_options"],
        total_transport_cost=optimisation["total_transport_cost"],
        persisted_rows=persisted,
    )


# ---------------------------------------------------------------------------
# Pipeline congestion endpoints
# ---------------------------------------------------------------------------


@router.post("/pipelines/congestion-train", response_model=PipelineTrainResponse)
async def train_pipeline_congestion(payload: PipelineTrainRequest) -> PipelineTrainResponse:
    flows = _ensure_non_empty_series(
        dal.get_fundamental_series(
            payload.flow_entity_id,
            payload.flow_variable,
            lookback_days=payload.lookback_days,
        ),
        "pipeline flows",
    )

    weather = _build_weather_frame(
        payload.weather_entity_id,
        payload.weather_variables,
        payload.lookback_days,
    )

    demand = _build_demand_series(
        payload.demand_entity_id,
        payload.demand_variable,
        payload.lookback_days,
    )

    if demand.empty:
        demand = flows.rolling(7).mean().fillna(method="bfill")

    training_result = pipeline_model.train_congestion_model(
        historical_flows=flows,
        weather_data=weather,
        demand_data=demand,
        maintenance_schedule=None,
    )

    if "error" in training_result:
        raise HTTPException(status_code=400, detail=training_result["error"])

    best_model = training_result["best_model"]
    best_model_payload = training_result["model_results"][best_model]
    pipeline_model.models[best_model] = best_model_payload

    persisted_rows = 0

    diagnostics = {
        "train_mae": best_model_payload.get("train_mae"),
        "test_mae": best_model_payload.get("test_mae"),
    }

    forecast_features = pipeline_model._prepare_congestion_features(
        flows,
        weather,
        demand,
        maintenance=None,
    ).dropna()

    tail_features = forecast_features.tail(payload.horizon_days)
    if not tail_features.empty:
        predictions = pipeline_model.predict_congestion(
            best_model,
            tail_features,
            forecast_horizon=len(tail_features),
        )

        risk = pipeline_model.assess_congestion_risk(predictions)
        risk_classes = risk["risk_classification"]

        forecast_records: List[Dict[str, Any]] = []
        for prediction_date, utilization in predictions.items():
            risk_tier = risk_classes.get(prediction_date, "low")
            forecast_records.append(
                {
                    "forecast_date": prediction_date.date(),
                    "pipeline_id": payload.pipeline_id,
                    "market": payload.market,
                    "segment": payload.segment,
                    "horizon_days": payload.horizon_days,
                    "utilization_forecast_pct": float(utilization * 100),
                    "congestion_probability": float(utilization),
                    "risk_score": _map_risk_to_score(risk_tier),
                    "risk_tier": risk_tier,
                    "alert_level": risk_tier,
                    "drivers": {
                        "train_mae": best_model_payload.get("train_mae"),
                        "test_mae": best_model_payload.get("test_mae"),
                    },
                }
            )

        persisted_rows = persistence.persist_pipeline_forecast(
            forecasts=forecast_records,
            model_version=payload.model_version,
            diagnostics=diagnostics,
        )

    metrics = {
        model: {
            "train_mae": data.get("train_mae"),
            "test_mae": data.get("test_mae"),
            "train_rmse": data.get("train_rmse"),
            "test_rmse": data.get("test_rmse"),
        }
        for model, data in training_result["model_results"].items()
    }

    formatted_importance: Dict[str, Any] = {}
    for model_name, importance_df in training_result.get("feature_importance", {}).items():
        try:
            formatted_importance[model_name] = importance_df.to_dict(orient="records")
        except Exception:
            formatted_importance[model_name] = {}

    return PipelineTrainResponse(
        best_model=best_model,
        metrics=metrics,
        feature_importance=formatted_importance,
        training_data_size=training_result["training_data_size"],
        persisted_rows=persisted_rows,
    )


@router.get("/pipelines/congestion-forecast")
async def get_pipeline_congestion_forecast(
    pipeline_id: str = Query(...),
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    limit: int = Query(30, gt=0, le=365),
) -> Dict[str, Any]:
    query = [
        "SELECT forecast_date, utilization_forecast_pct, risk_tier, alert_level, risk_score",
        " FROM ch.pipeline_congestion_forecast",
        " WHERE pipeline_id = %(pipeline_id)s",
    ]
    params: Dict[str, Any] = {"pipeline_id": pipeline_id}

    if start:
        query.append("   AND forecast_date >= %(start)s")
        params["start"] = start
    if end:
        query.append("   AND forecast_date <= %(end)s")
        params["end"] = end

    query.append(" ORDER BY forecast_date DESC LIMIT %(limit)s")
    params["limit"] = limit

    rows = persistence.client.execute("".join(query), params)
    forecasts = [
        {
            "forecast_date": row[0],
            "utilization_forecast_pct": row[1],
            "risk_tier": row[2],
            "alert_level": row[3],
            "risk_score": row[4],
        }
        for row in rows
    ]

    return {"pipeline_id": pipeline_id, "forecasts": forecasts}


@router.get("/pipelines/congestion-alerts")
async def get_pipeline_alerts(
    pipeline_id: str = Query(...),
    lookahead_days: int = Query(7, gt=0, le=90),
) -> Dict[str, Any]:
    query = (
        "SELECT forecast_date, utilization_forecast_pct, risk_tier "
        "FROM ch.pipeline_congestion_forecast "
        "WHERE pipeline_id = %(pipeline_id)s "
        "  AND forecast_date >= today() "
        "  AND forecast_date <= today() + %(lookahead)s "
        "ORDER BY forecast_date ASC"
    )
    rows = persistence.client.execute(query, {"pipeline_id": pipeline_id, "lookahead": lookahead_days})
    if not rows:
        return {"pipeline_id": pipeline_id, "alerts": []}

    series_index = [pd.Timestamp(row[0]) for row in rows]
    utilization = pd.Series([row[1] / 100 for row in rows], index=series_index)
    alerts = pipeline_model.generate_congestion_alerts(utilization)

    return {"pipeline_id": pipeline_id, "alerts": alerts}


# ---------------------------------------------------------------------------
# Seasonal demand endpoints
# ---------------------------------------------------------------------------


@router.post("/demand/seasonal-forecast", response_model=SeasonalForecastResponse)
async def forecast_seasonal_demand(payload: SeasonalForecastRequest) -> SeasonalForecastResponse:
    historical = _ensure_non_empty_series(
        dal.get_fundamental_series(
            payload.historical_entity_id,
            payload.historical_variable,
            lookback_days=max(payload.horizon_days * 2, 365),
        ),
        "historical demand",
    )

    temperatures = _ensure_non_empty_series(
        dal.get_weather_series(
            payload.weather_entity_id,
            payload.weather_variable,
            lookback_days=payload.horizon_days,
        ),
        "temperature forecast",
    )

    demand_model = seasonal_model._build_demand_model(historical, payload.region)
    base_forecast = seasonal_model._generate_base_forecast(historical, payload.horizon_days, demand_model)
    degree_days = seasonal_model._calculate_degree_days_from_temperature(temperatures)
    weather_adjusted = seasonal_model._apply_weather_adjustments(base_forecast, degree_days, payload.region)

    economic_adjustment = None
    if payload.economic_indicators:
        econ_series: Dict[str, pd.Series] = {}
        for label, identifier in payload.economic_indicators.items():
            entity, variable = identifier.split(":") if ":" in identifier else (payload.historical_entity_id, identifier)
            econ_series[label] = dal.get_fundamental_series(entity, variable, lookback_days=365)
        economic_adjustment = econ_series
        final_forecast = seasonal_model._apply_economic_adjustments(weather_adjusted, econ_series, payload.region)
    else:
        final_forecast = weather_adjusted

    forecast_dates = list(base_forecast.index[:payload.horizon_days])
    records: List[Dict[str, Any]] = []

    historical_std = float(historical.std()) if len(historical) > 1 else 0.0

    for forecast_day in forecast_dates:
        base_value = float(base_forecast.loc[forecast_day])
        weather_value = float(weather_adjusted.loc[forecast_day])
        final_value = float(final_forecast.loc[forecast_day])
        records.append(
            {
                "forecast_date": forecast_day.date(),
                "region": payload.region,
                "sector": payload.sector,
                "scenario_id": payload.scenario_id,
                "base_demand_mw": base_value,
                "weather_adjustment_mw": weather_value - base_value,
                "economic_adjustment_mw": (final_value - weather_value) if payload.economic_indicators else None,
                "holiday_adjustment_mw": None,
                "final_forecast_mw": final_value,
                "peak_risk_score": min(final_value / (historical.max() or final_value), 1.0),
                "confidence_low_mw": max(final_value - 1.96 * historical_std, 0.0),
                "confidence_high_mw": final_value + 1.96 * historical_std,
            }
        )

    persisted = persistence.persist_seasonal_demand(
        records=records,
        model_version=payload.model_version,
        diagnostics={
            "economic_indicators": list(payload.economic_indicators.keys()) if payload.economic_indicators else [],
            "region": payload.region,
        },
    )

    return SeasonalForecastResponse(records=records, persisted_rows=persisted)


@router.get("/demand/peak-risk", response_model=PeakRiskResponse)
async def assess_peak_demand(
    region: str = Query(...),
    historical_entity_id: str = Query(...),
    historical_variable: str = Query(...),
    scenarios: List[str] = Query(..., description="List of scenario payloads as JSON strings"),
) -> PeakRiskResponse:
    historical = _ensure_non_empty_series(
        dal.get_fundamental_series(
            historical_entity_id,
            historical_variable,
            lookback_days=1460,
        ),
        "historical demand",
    )

    scenario_payloads: List[Dict[str, Any]] = []
    for raw in scenarios:
        try:
            scenario_payloads.append(json.loads(raw))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid scenario payload: {raw}") from exc

    peak = seasonal_model.predict_peak_demand(historical, scenario_payloads)

    return PeakRiskResponse(
        historical_peak=peak.get("historical_peak", 0.0),
        extreme_weather_peak=peak.get("extreme_weather_peak", 0.0),
        statistical_peak=peak.get("statistical_peak", 0.0),
        return_period_peak=peak.get("return_period_peak", 0.0),
        details={
            "return_period_years": peak.get("return_period_years"),
            "avg_extreme_multiplier": peak.get("avg_extreme_multiplier"),
            "extreme_scenarios_analyzed": peak.get("extreme_scenarios_analyzed"),
        },
    )


__all__ = ["router"]
