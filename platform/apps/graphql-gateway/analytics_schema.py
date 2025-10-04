"""GraphQL schema extensions for gas & coal analytics."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import strawberry
from clickhouse_driver import Client
from strawberry.types import Info

from platform.shared.cache_utils import cache_sync


logger = logging.getLogger(__name__)


@dataclass
class ClickHouseResult:
    data: list
    columns: list


def _execute(
    client: Client, query: str, parameters: dict, settings: Optional[Dict[str, Any]] = None
) -> ClickHouseResult:
    execute_kwargs: Dict[str, Any] = {"with_column_types": True}
    if settings:
        execute_kwargs["settings"] = settings
    data, column_types = client.execute(query, parameters, **execute_kwargs)
    columns = [col for col, _ in column_types]
    return ClickHouseResult(data, columns)


def _rows_to_dicts(result: ClickHouseResult) -> List[dict]:
    return [dict(zip(result.columns, row)) for row in result.data]


def _maybe_json(value: Optional[Any]) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


_ANALYTICS_CACHE_NAMESPACE = "analytics"
_DEFAULT_CORRELATION_LIMIT = 500
_MAX_CORRELATION_LIMIT = 1_000
_DEFAULT_CORRELATION_WINDOW_DAYS = 90
_MAX_CORRELATION_WINDOW_DAYS = 365
_DEFAULT_SEASONALITY_WINDOW_DAYS = 365
_MAX_SEASONALITY_WINDOW_DAYS = 730
_DEFAULT_VOL_WINDOW_DAYS = 365
_MAX_VOL_WINDOW_DAYS = 730


def _clamp_limit(value: Optional[int], default: int, maximum: int) -> int:
    if value is None:
        return default
    if value < 1:
        return 1
    return min(value, maximum)


def _normalize_instruments(instruments: Optional[Iterable[str]]) -> List[str]:
    if not instruments:
        return []
    normalized = sorted({instrument.strip().upper() for instrument in instruments if instrument})
    return [instrument for instrument in normalized if instrument]


def _ensure_date_range(
    start: Optional[date],
    end: Optional[date],
    default_days: int,
    max_days: int,
) -> Tuple[Optional[date], Optional[date]]:
    if start and end and start > end:
        raise ValueError("start must be <= end")

    if not end:
        end = date.today()

    if not start:
        start = end - timedelta(days=default_days)

    span = (end - start).days if start and end else None
    if span is not None and span > max_days:
        start = end - timedelta(days=max_days)

    return start, end


def _to_iso(value: Optional[date]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _date_from_iso(value: str) -> date:
    return date.fromisoformat(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_str_param(
    params: Dict[str, Any], key: str, *, required: bool = False, max_length: int = 200
) -> Optional[str]:
    value = params.get(key)
    if value is None:
        if required:
            raise ValueError(f"'{key}' parameter is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string")
    trimmed = value.strip()
    if required and not trimmed:
        raise ValueError(f"'{key}' cannot be blank")
    if len(trimmed) > max_length:
        raise ValueError(f"'{key}' exceeds maximum length {max_length}")
    return trimmed


def _extract_date_param(params: Dict[str, Any], key: str) -> Optional[date]:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"'{key}' must be an ISO date (YYYY-MM-DD)") from exc
    raise ValueError(f"'{key}' must be an ISO date string")


@strawberry.type
class StorageScheduleEntry:
    date: date
    action: str
    volume_mmbtu: float
    inventory_mmbtu: float
    price: float
    net_cash_flow: float


@strawberry.type
class StorageArbitrageOutput:
    as_of_date: date
    hub: str
    region: Optional[str]
    expected_storage_value: float
    breakeven_spread: Optional[float]
    schedule: List[StorageScheduleEntry]
    cost_parameters: strawberry.scalars.JSON
    constraint_summary: strawberry.scalars.JSON
    diagnostics: strawberry.scalars.JSON


@strawberry.type
class WeatherImpactOutput:
    as_of_date: date
    entity_id: str
    coef_type: str
    coefficient: float
    r2: Optional[float]
    window: str
    diagnostics: strawberry.scalars.JSON


@strawberry.type
class SwitchingEconomicsOutput:
    as_of_date: date
    region: str
    coal_cost_mwh: float
    gas_cost_mwh: float
    co2_price: float
    breakeven_gas_price: float
    switch_share: float
    diagnostics: strawberry.scalars.JSON


@strawberry.type
class BasisModelOutput:
    as_of_date: date
    hub: str
    predicted_basis: float
    actual_basis: Optional[float]
    method: str
    diagnostics: strawberry.scalars.JSON
    feature_snapshot: strawberry.scalars.JSON


@strawberry.type
class LNGRoutingOption:
    route_id: str
    export_terminal: str
    import_terminal: str
    vessel_type: Optional[str]
    cargo_size_bcf: float
    vessel_speed_knots: float
    fuel_price_usd_per_tonne: float
    distance_nm: float
    voyage_time_days: float
    fuel_consumption_tonnes: float
    fuel_cost_usd: float
    charter_cost_usd: float
    port_cost_usd: Optional[float]
    total_cost_usd: float
    cost_per_mmbtu_usd: float
    is_optimal_route: bool
    assumptions: Optional[strawberry.scalars.JSON]
    diagnostics: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]


@strawberry.type
class LNGRoutingResult:
    as_of_date: date
    options: List[LNGRoutingOption]
    metadata: strawberry.scalars.JSON


@strawberry.type
class CoalRouteBreakdown:
    bunker_cost_usd: Optional[float]
    port_fees_usd: Optional[float]
    congestion_premium_usd: Optional[float]
    carbon_cost_usd: Optional[float]
    demurrage_cost_usd: Optional[float]
    rail_cost_usd: Optional[float]
    truck_cost_usd: Optional[float]


@strawberry.type
class CoalTransportCost:
    as_of_month: date
    route_id: str
    origin_region: str
    destination_region: str
    transport_mode: str
    vessel_type: Optional[str]
    cargo_tonnes: float
    fuel_price_usd_per_tonne: float
    freight_cost_usd: float
    total_cost_usd: float
    currency: str
    breakdown: CoalRouteBreakdown
    assumptions: Optional[strawberry.scalars.JSON]
    diagnostics: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]


@strawberry.type
class PipelineCongestionPoint:
    forecast_date: date
    utilization_forecast_pct: float
    congestion_probability: float
    risk_score: float
    risk_tier: str
    alert_level: Optional[str]
    drivers: Optional[strawberry.scalars.JSON]
    diagnostics: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]


@strawberry.type
class PipelineCongestionAlert:
    date: date
    utilization_forecast_pct: float
    risk_tier: str
    alert_level: Optional[str]
    message: str


@strawberry.type
class CrackOptimizationResult:
    as_of_date: date
    region: str
    refinery_id: Optional[str]
    crack_type: str
    crude_code: str
    gasoline_price: float
    diesel_price: float
    jet_price: Optional[float]
    crack_spread: float
    margin_per_bbl: float
    optimal_yields: Optional[strawberry.scalars.JSON]
    constraints: Optional[strawberry.scalars.JSON]
    diagnostics: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class RefineryYieldResult:
    as_of_date: date
    crude_type: str
    process_config: Optional[strawberry.scalars.JSON]
    yields: Optional[strawberry.scalars.JSON]
    value_per_bbl: float
    operating_cost: float
    net_value: float
    diagnostics: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class ProductElasticityResult:
    as_of_date: date
    product: str
    region: Optional[str]
    method: str
    elasticity: float
    r_squared: Optional[float]
    own_or_cross: str
    product_pair: Optional[str]
    data_points: int
    diagnostics: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class TransportFuelSubstitutionMetric:
    as_of_date: date
    region: str
    metric: str
    value: float
    details: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class RINPriceForecastPoint:
    as_of_date: date
    rin_category: str
    horizon_days: int
    forecast_date: date
    forecast_price: float
    std: Optional[float]
    drivers: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class BiodieselSpreadResult:
    as_of_date: date
    region: Optional[str]
    mean_gross_spread: float
    mean_net_spread: float
    spread_volatility: float
    arbitrage_opportunities: int
    diagnostics: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class CarbonIntensityResult:
    as_of_date: date
    fuel_type: str
    pathway: str
    total_ci: float
    base_emissions: float
    transport_emissions: float
    land_use_emissions: float
    ci_per_mj: float
    assumptions: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class RenewablesPolicyImpactResult:
    as_of_date: date
    policy: str
    entity: str
    metric: str
    value: float
    details: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class CarbonPriceForecastPoint:
    as_of_date: date
    market: str
    horizon_days: int
    forecast_date: date
    forecast_price: float
    std: Optional[float]
    drivers: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class ComplianceCostResult:
    as_of_date: date
    market: str
    sector: str
    total_emissions: float
    average_price: float
    cost_per_tonne: float
    total_compliance_cost: float
    details: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class CarbonLeakageRisk:
    as_of_date: date
    sector: str
    domestic_price: float
    international_price: float
    price_differential: float
    trade_exposure: float
    emissions_intensity: float
    leakage_risk_score: float
    risk_level: str
    details: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class DecarbonizationPathwayResult:
    as_of_date: date
    sector: str
    policy_scenario: str
    target_year: int
    annual_reduction_rate: float
    cumulative_emissions: float
    target_achieved: bool
    emissions_trajectory: Optional[strawberry.scalars.JSON]
    technology_analysis: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class RenewableAdoptionPoint:
    as_of_date: date
    technology: str
    forecast_year: date
    capacity_gw: float
    policy_support: float
    economic_multipliers: Optional[strawberry.scalars.JSON]
    assumptions: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class StrandedAssetRisk:
    as_of_date: date
    asset_type: str
    asset_value: float
    carbon_cost_pv: float
    stranded_value: float
    stranded_ratio: float
    remaining_lifetime: int
    risk_level: str
    details: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class PolicyScenarioImpact:
    as_of_date: date
    scenario: str
    entity: str
    metric: str
    value: float
    details: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]
    created_at: Optional[datetime]


@strawberry.type
class DemandForecastPoint:
    forecast_date: date
    region: str
    sector: Optional[str]
    scenario_id: str
    base_demand_mw: float
    weather_adjustment_mw: Optional[float]
    economic_adjustment_mw: Optional[float]
    holiday_adjustment_mw: Optional[float]
    final_forecast_mw: float
    peak_risk_score: Optional[float]
    confidence_low_mw: Optional[float]
    confidence_high_mw: Optional[float]
    diagnostics: Optional[strawberry.scalars.JSON]
    model_version: Optional[str]


@strawberry.type
class PeakDemandAssessment:
    forecast_peak_mw: Optional[float]
    average_peak_risk: Optional[float]
    observations: strawberry.scalars.JSON


@strawberry.type
class SeasonalDemandResult:
    region: str
    scenario_id: str
    points: List[DemandForecastPoint]
    peak_assessment: PeakDemandAssessment


@strawberry.type
class PortfolioOptimizationRun:
    run_id: strawberry.ID
    portfolio_id: str
    as_of_date: date
    method: str
    params: Optional[strawberry.scalars.JSON]
    weights: strawberry.scalars.JSON
    metrics: strawberry.scalars.JSON
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


@strawberry.type
class PortfolioRiskMetric:
    as_of_date: date
    portfolio_id: str
    run_id: strawberry.ID
    volatility: float
    variance: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    beta: Optional[float]
    diversification_benefit: float
    exposures: strawberry.scalars.JSON
    created_at: Optional[datetime]


@strawberry.type
class PortfolioStressResult:
    as_of_date: date
    portfolio_id: str
    run_id: strawberry.ID
    scenario_id: str
    scenario: strawberry.scalars.JSON
    metrics: strawberry.scalars.JSON
    severity: str
    probability: float
    created_at: Optional[datetime]


@strawberry.type
class ArbitrageOpportunity:
    as_of_date: date
    commodity1: str
    commodity2: str
    instrument1: str
    instrument2: str
    mean_spread: float
    spread_volatility: float
    transport_cost: float
    storage_cost: float
    net_profit: float
    direction: str
    confidence: float
    period: Optional[str]
    metadata: Optional[strawberry.scalars.JSON]
    created_at: Optional[datetime]


@strawberry.type
class CorrelationPair:
    date: date
    instrument1: str
    instrument2: str
    correlation: float
    sample_count: int


@strawberry.type
class CorrelationMatrix:
    date: date
    instruments: List[str]
    coefficients: strawberry.scalars.JSON


@strawberry.type
class VolatilityPoint:
    as_of_date: date
    instrument_id: str
    vol_30d: Optional[float]
    vol_90d: Optional[float]
    vol_365d: Optional[float]


@strawberry.type
class DecompositionPoint:
    snapshot_date: date
    instrument_id: str
    method: str
    trend: Optional[float]
    seasonal: Optional[float]
    residual: Optional[float]


@strawberry.enum
class ResearchQueryId(str):
    LIST_NOTEBOOKS = "LIST_NOTEBOOKS"
    LIST_EXPERIMENTS = "LIST_EXPERIMENTS"
    EXPERIMENTS_BY_MODEL = "EXPERIMENTS_BY_MODEL"
    VOLATILITY_REGIME_SHARE = "VOLATILITY_REGIME_SHARE"
    DECOMPOSITION_SUMMARY = "DECOMPOSITION_SUMMARY"


@strawberry.input
class ResearchQueryInput:
    query_id: ResearchQueryId
    params: Optional[strawberry.scalars.JSON] = None
    limit: Optional[int] = None


@strawberry.type
class ResearchRow:
    columns: List[str]
    values: List[strawberry.scalars.JSON]


@dataclass
class ResearchQueryTemplate:
    description: str
    default_limit: int
    max_limit: int
    builder: Callable[[Dict[str, Any], int], Tuple[str, Dict[str, Any], Dict[str, Any]]]

    def describe(self) -> str:
        return self.description


def _correlation_pairs_cache_key(_: Client, filters: Dict[str, Any]) -> str:
    payload = {
        "start": _to_iso(filters.get("start")),
        "end": _to_iso(filters.get("end")),
        "instruments": filters.get("instruments") or [],
        "min_samples": filters.get("min_samples"),
        "limit": filters.get("limit"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@cache_sync(namespace=_ANALYTICS_CACHE_NAMESPACE, key_func=_correlation_pairs_cache_key, ttl=90)
def _fetch_correlation_pairs(client: Client, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "limit": filters["limit"],
        "min_samples": filters["min_samples"],
    }
    conditions = ["sample_count >= %(min_samples)s"]

    if filters.get("start"):
        params["start"] = filters["start"]
        conditions.append("date >= %(start)s")
    if filters.get("end"):
        params["end"] = filters["end"]
        conditions.append("date <= %(end)s")
    instruments = filters.get("instruments")
    if instruments:
        params["instrument_filter"] = tuple(instruments)
        conditions.append(
            "(instrument1 IN %(instrument_filter)s OR instrument2 IN %(instrument_filter)s)"
        )

    where_clause = " AND ".join(conditions) if conditions else "1 = 1"
    query = f"""
        SELECT
            date,
            instrument1,
            instrument2,
            correlation_coefficient AS correlation,
            sample_count
        FROM ch.commodity_correlations
        WHERE {where_clause}
        ORDER BY date DESC, instrument1, instrument2
        LIMIT %(limit)s
    """
    result = _execute(client, query, params)
    rows = _rows_to_dicts(result)
    return [
        {
            "date": _to_iso(row["date"]),
            "instrument1": row["instrument1"],
            "instrument2": row["instrument2"],
            "correlation": row["correlation"],
            "sample_count": row["sample_count"],
        }
        for row in rows
    ]


def _correlation_matrix_cache_key(_: Client, filters: Dict[str, Any]) -> str:
    payload = {
        "date": _to_iso(filters.get("date")),
        "instruments": filters.get("instruments") or [],
        "min_samples": filters.get("min_samples"),
        "limit": filters.get("limit"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@cache_sync(namespace=_ANALYTICS_CACHE_NAMESPACE, key_func=_correlation_matrix_cache_key, ttl=120)
def _fetch_correlation_matrix(client: Client, filters: Dict[str, Any]) -> Dict[str, Any]:
    instruments: List[str] = filters["instruments"]
    if not instruments:
        return {"date": None, "instruments": [], "pairs": []}

    min_samples = filters["min_samples"]
    limit = filters["limit"]

    base_params: Dict[str, Any] = {
        "instrument_filter": tuple(instruments),
        "min_samples": min_samples,
    }

    target_date: Optional[date] = filters.get("date")
    if not target_date:
        latest_query = """
            SELECT max(date)
            FROM ch.commodity_correlations
            WHERE instrument1 IN %(instrument_filter)s
              AND instrument2 IN %(instrument_filter)s
              AND sample_count >= %(min_samples)s
        """
        latest_result = _execute(client, latest_query, base_params)
        if not latest_result.data or latest_result.data[0][0] is None:
            return {"date": None, "instruments": instruments, "pairs": []}
        target_date = latest_result.data[0][0]

    params = dict(base_params)
    params["target_date"] = target_date
    max_pairs = len(instruments) * len(instruments)
    params["limit"] = min(limit, max_pairs if max_pairs else limit)

    matrix_query = """
        SELECT
            date,
            instrument1,
            instrument2,
            correlation_coefficient AS correlation
        FROM ch.commodity_correlations
        WHERE date = %(target_date)s
          AND instrument1 IN %(instrument_filter)s
          AND instrument2 IN %(instrument_filter)s
          AND sample_count >= %(min_samples)s
        ORDER BY instrument1, instrument2
        LIMIT %(limit)s
    """
    result = _execute(client, matrix_query, params)
    rows = _rows_to_dicts(result)
    return {
        "date": _to_iso(target_date),
        "instruments": instruments,
        "pairs": [
            {
                "instrument1": row["instrument1"],
                "instrument2": row["instrument2"],
                "correlation": row["correlation"],
            }
            for row in rows
        ],
    }


def _volatility_surface_cache_key(_: Client, filters: Dict[str, Any]) -> str:
    payload = {
        "start": _to_iso(filters.get("start")),
        "end": _to_iso(filters.get("end")),
        "instruments": filters.get("instruments") or [],
        "limit": filters.get("limit"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@cache_sync(namespace=_ANALYTICS_CACHE_NAMESPACE, key_func=_volatility_surface_cache_key, ttl=60)
def _fetch_volatility_surface(client: Client, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": filters["limit"]}
    conditions = ["1 = 1"]

    if filters.get("start"):
        params["start"] = filters["start"]
        conditions.append("as_of_date >= %(start)s")
    if filters.get("end"):
        params["end"] = filters["end"]
        conditions.append("as_of_date <= %(end)s")

    instruments = filters.get("instruments")
    if instruments:
        params["instrument_filter"] = tuple(instruments)
        conditions.append("instrument_id IN %(instrument_filter)s")

    order_clause = "ORDER BY as_of_date ASC, instrument_id"
    query = f"""
        SELECT
            as_of_date,
            instrument_id,
            volatility_30d AS vol_30d,
            volatility_90d AS vol_90d,
            volatility_365d AS vol_365d
        FROM ch.commodity_volatility_surface
        WHERE {' AND '.join(conditions)}
        {order_clause}
        LIMIT %(limit)s
    """
    result = _execute(client, query, params)
    rows = _rows_to_dicts(result)
    return [
        {
            "as_of_date": _to_iso(row["as_of_date"]),
            "instrument_id": row["instrument_id"],
            "vol_30d": row.get("vol_30d"),
            "vol_90d": row.get("vol_90d"),
            "vol_365d": row.get("vol_365d"),
        }
        for row in rows
    ]


def _seasonality_cache_key(_: Client, filters: Dict[str, Any]) -> str:
    payload = {
        "instrument_id": filters.get("instrument_id"),
        "method": filters.get("method"),
        "start": _to_iso(filters.get("start")),
        "end": _to_iso(filters.get("end")),
        "limit": filters.get("limit"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@cache_sync(namespace=_ANALYTICS_CACHE_NAMESPACE, key_func=_seasonality_cache_key, ttl=120)
def _fetch_seasonality_points(client: Client, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "instrument_id": filters["instrument_id"],
        "method": filters["method"],
        "limit": filters["limit"],
    }
    conditions = [
        "instrument_id = %(instrument_id)s",
        "method = %(method)s",
    ]

    if filters.get("start"):
        params["start"] = filters["start"]
        conditions.append("snapshot_date >= %(start)s")
    if filters.get("end"):
        params["end"] = filters["end"]
        conditions.append("snapshot_date <= %(end)s")

    query = f"""
        SELECT
            snapshot_date,
            instrument_id,
            method,
            trend,
            seasonal,
            residual
        FROM ch.commodity_decomposition
        WHERE {' AND '.join(conditions)}
        ORDER BY snapshot_date ASC
        LIMIT %(limit)s
    """
    result = _execute(client, query, params)
    rows = _rows_to_dicts(result)
    return [
        {
            "snapshot_date": _to_iso(row["snapshot_date"]),
            "instrument_id": row["instrument_id"],
            "method": row["method"],
            "trend": row.get("trend"),
            "seasonal": row.get("seasonal"),
            "residual": row.get("residual"),
        }
        for row in rows
    ]


def _seasonality_latest_cache_key(_: Client, filters: Dict[str, Any]) -> str:
    payload = {
        "instrument_id": filters.get("instrument_id"),
        "method": filters.get("method"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@cache_sync(namespace=_ANALYTICS_CACHE_NAMESPACE, key_func=_seasonality_latest_cache_key, ttl=120)
def _fetch_seasonality_latest(client: Client, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "instrument_id": filters["instrument_id"],
        "method": filters["method"],
    }
    query = """
        SELECT
            snapshot_date,
            instrument_id,
            method,
            trend,
            seasonal,
            residual
        FROM ch.commodity_decomposition
        WHERE instrument_id = %(instrument_id)s
          AND method = %(method)s
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    result = _execute(client, query, params)
    rows = _rows_to_dicts(result)
    if not rows:
        return None
    row = rows[0]
    return {
        "snapshot_date": _to_iso(row["snapshot_date"]),
        "instrument_id": row["instrument_id"],
        "method": row["method"],
        "trend": row.get("trend"),
        "seasonal": row.get("seasonal"),
        "residual": row.get("residual"),
    }


def _research_query_cache_key(
    _: Client, query_id: str, sql: str, params: Dict[str, Any], key_params: Dict[str, Any]
) -> str:
    del sql, params  # unused for cache key; rely on structured params instead
    payload = {"query_id": query_id, "params": key_params}
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@cache_sync(namespace=_ANALYTICS_CACHE_NAMESPACE, key_func=_research_query_cache_key, ttl=60)
def _run_research_query_cached(
    client: Client,
    query_id: str,
    sql: str,
    params: Dict[str, Any],
    key_params: Dict[str, Any],
) -> Dict[str, Any]:
    limit = key_params.get("limit", 200)
    settings = {
        "readonly": 1,
        "max_execution_time": 5,
        "max_result_rows": limit,
        "max_result_bytes": 10_000_000,
    }
    result = _execute(client, sql, params, settings=settings)
    rows = [
        [_json_safe(value) for value in row]
        for row in result.data
    ]
    return {
        "columns": result.columns,
        "rows": rows,
        "meta": key_params,
        "query_id": query_id,
    }


def _build_list_notebooks(params: Dict[str, Any], limit: int) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    status = _extract_str_param(params, "status", max_length=32)
    author = _extract_str_param(params, "author", max_length=64)

    query_params: Dict[str, Any] = {"limit": limit}
    conditions: List[str] = ["1 = 1"]

    if status:
        query_params["status"] = status
        conditions.append("status = %(status)s")
    if author:
        query_params["author"] = author
        conditions.append("author = %(author)s")

    sql = f"""
        SELECT
            notebook_id,
            title,
            author,
            status,
            created_at,
            executed_at
        FROM ch.research_notebooks
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT %(limit)s
    """
    key_params = {"status": status, "author": author, "limit": limit}
    return sql, query_params, key_params


def _build_list_experiments(params: Dict[str, Any], limit: int) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    status = _extract_str_param(params, "status", max_length=32)
    dataset = _extract_str_param(params, "dataset", max_length=128)

    query_params: Dict[str, Any] = {"limit": limit}
    conditions: List[str] = ["1 = 1"]

    if status:
        query_params["status"] = status
        conditions.append("status = %(status)s")
    if dataset:
        query_params["dataset"] = dataset
        conditions.append("dataset = %(dataset)s")

    sql = f"""
        SELECT
            experiment_id,
            name,
            model_type,
            dataset,
            status,
            started_at,
            completed_at
        FROM ch.research_experiments
        WHERE {' AND '.join(conditions)}
        ORDER BY started_at DESC
        LIMIT %(limit)s
    """
    key_params = {"status": status, "dataset": dataset, "limit": limit}
    return sql, query_params, key_params


def _build_experiments_by_model(
    params: Dict[str, Any], limit: int
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    model = _extract_str_param(params, "model", required=True, max_length=64)
    if model is None:  # for type checker, though required ensures not None
        raise ValueError("'model' parameter is required")

    status = _extract_str_param(params, "status", max_length=32)
    start = _extract_date_param(params, "start")
    end = _extract_date_param(params, "end")

    if start and end and start > end:
        raise ValueError("'start' must be on or before 'end'")

    query_params: Dict[str, Any] = {
        "model": model,
        "limit": limit,
    }
    conditions: List[str] = ["model_type = %(model)s"]

    if status:
        query_params["status"] = status
        conditions.append("status = %(status)s")
    if start:
        query_params["start"] = start
        conditions.append("toDate(started_at) >= %(start)s")
    if end:
        query_params["end"] = end
        conditions.append("toDate(started_at) <= %(end)s")

    sql = f"""
        SELECT
            experiment_id,
            name,
            model_type,
            dataset,
            status,
            started_at,
            completed_at,
            mlflow_run_id
        FROM ch.research_experiments
        WHERE {' AND '.join(conditions)}
        ORDER BY started_at DESC
        LIMIT %(limit)s
    """
    key_params = {
        "model": model,
        "status": status,
        "start": _to_iso(start),
        "end": _to_iso(end),
        "limit": limit,
    }
    return sql, query_params, key_params


def _build_volatility_regime_share(
    params: Dict[str, Any], limit: int
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    instrument_id = _extract_str_param(params, "instrument_id", required=True, max_length=64)
    if instrument_id is None:
        raise ValueError("'instrument_id' parameter is required")

    regime = _extract_str_param(params, "regime", max_length=32)
    start = _extract_date_param(params, "start")
    end = _extract_date_param(params, "end")

    if start and end and start > end:
        raise ValueError("'start' must be on or before 'end'")

    query_params: Dict[str, Any] = {
        "instrument_id": instrument_id,
        "limit": limit,
    }
    conditions: List[str] = ["instrument_id = %(instrument_id)s"]

    if regime:
        query_params["regime"] = regime
        conditions.append("regime_label = %(regime)s")
    if start:
        query_params["start"] = start
        conditions.append("month >= %(start)s")
    if end:
        query_params["end"] = end
        conditions.append("month <= %(end)s")

    sql = f"""
        SELECT
            month,
            instrument_id,
            regime_label,
            observation_count
        FROM ch.mv_volatility_regime_share_monthly
        WHERE {' AND '.join(conditions)}
        ORDER BY month DESC, regime_label
        LIMIT %(limit)s
    """
    key_params = {
        "instrument_id": instrument_id,
        "regime": regime,
        "start": _to_iso(start),
        "end": _to_iso(end),
        "limit": limit,
    }
    return sql, query_params, key_params


def _build_decomposition_summary(
    params: Dict[str, Any], limit: int
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    instrument_id = _extract_str_param(params, "instrument_id", required=True, max_length=64)
    if instrument_id is None:
        raise ValueError("'instrument_id' parameter is required")

    method = _extract_str_param(params, "method", max_length=32) or "stl"
    start = _extract_date_param(params, "start")
    end = _extract_date_param(params, "end")

    if start and end and start > end:
        raise ValueError("'start' must be on or before 'end'")

    query_params: Dict[str, Any] = {
        "instrument_id": instrument_id,
        "method": method,
        "limit": limit,
    }
    conditions: List[str] = [
        "instrument_id = %(instrument_id)s",
        "method = %(method)s",
    ]

    if start:
        query_params["start"] = start
        conditions.append("snapshot_date >= %(start)s")
    if end:
        query_params["end"] = end
        conditions.append("snapshot_date <= %(end)s")

    sql = f"""
        SELECT
            snapshot_date,
            instrument_id,
            method,
            avgMerge(avg_trend_state) AS avg_trend,
            avgMerge(seasonal_intensity_state) AS seasonal_intensity
        FROM ch.mv_commodity_decomposition_daily
        WHERE {' AND '.join(conditions)}
        GROUP BY snapshot_date, instrument_id, method
        ORDER BY snapshot_date DESC
        LIMIT %(limit)s
    """
    key_params = {
        "instrument_id": instrument_id,
        "method": method,
        "start": _to_iso(start),
        "end": _to_iso(end),
        "limit": limit,
    }
    return sql, query_params, key_params


_RESEARCH_TEMPLATES: Dict[ResearchQueryId, ResearchQueryTemplate] = {
    ResearchQueryId.LIST_NOTEBOOKS: ResearchQueryTemplate(
        description="LIST_NOTEBOOKS(status: String, author: String, limit<=200)",
        default_limit=50,
        max_limit=200,
        builder=_build_list_notebooks,
    ),
    ResearchQueryId.LIST_EXPERIMENTS: ResearchQueryTemplate(
        description="LIST_EXPERIMENTS(status: String, dataset: String, limit<=200)",
        default_limit=50,
        max_limit=200,
        builder=_build_list_experiments,
    ),
    ResearchQueryId.EXPERIMENTS_BY_MODEL: ResearchQueryTemplate(
        description="EXPERIMENTS_BY_MODEL(model: String!, status: String, start: Date, end: Date, limit<=200)",
        default_limit=100,
        max_limit=200,
        builder=_build_experiments_by_model,
    ),
    ResearchQueryId.VOLATILITY_REGIME_SHARE: ResearchQueryTemplate(
        description="VOLATILITY_REGIME_SHARE(instrument_id: String!, regime: String, start: Date, end: Date, limit<=200)",
        default_limit=120,
        max_limit=240,
        builder=_build_volatility_regime_share,
    ),
    ResearchQueryId.DECOMPOSITION_SUMMARY: ResearchQueryTemplate(
        description="DECOMPOSITION_SUMMARY(instrument_id: String!, method: String = 'stl', start: Date, end: Date, limit<=200)",
        default_limit=120,
        max_limit=240,
        builder=_build_decomposition_summary,
    ),
}


class AnalyticsQuery:
    """Analytics fields for gas & coal workflows."""

    @strawberry.field
    def correlation_pairs(
        self,
        info: Info,
        instruments: Optional[List[str]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        min_samples: int = 30,
        limit: int = 500,
    ) -> List[CorrelationPair]:
        client: Client = info.context["ch_client"]
        normalized = _normalize_instruments(instruments)
        min_samples_value = max(1, min(min_samples, 10_000))
        limit_value = _clamp_limit(limit, _DEFAULT_CORRELATION_LIMIT, _MAX_CORRELATION_LIMIT)
        start, end = _ensure_date_range(
            start,
            end,
            _DEFAULT_CORRELATION_WINDOW_DAYS,
            _MAX_CORRELATION_WINDOW_DAYS,
        )

        filters = {
            "start": start,
            "end": end,
            "instruments": normalized,
            "min_samples": min_samples_value,
            "limit": limit_value,
        }

        timer = time.perf_counter()
        cached_rows = _fetch_correlation_pairs(client, filters)
        elapsed = time.perf_counter() - timer

        pairs: List[CorrelationPair] = []
        for row in cached_rows:
            raw_date = row.get("date")
            if not raw_date:
                continue
            pairs.append(
                CorrelationPair(
                    date=_date_from_iso(raw_date),
                    instrument1=row["instrument1"],
                    instrument2=row["instrument2"],
                    correlation=row["correlation"],
                    sample_count=row["sample_count"],
                )
            )

        logger.info(
            "analytics.correlation_pairs instruments=%s rows=%d elapsed=%.3fs",
            ",".join(normalized) if normalized else "ALL",
            len(pairs),
            elapsed,
        )
        return pairs

    @strawberry.field
    def correlation_matrix(
        self,
        info: Info,
        instruments: List[str],
        date: Optional[date] = None,
        min_samples: int = 30,
        limit: int = 2500,
    ) -> Optional[CorrelationMatrix]:
        normalized = _normalize_instruments(instruments)
        if not normalized:
            raise ValueError("At least one instrument must be provided")
        if len(normalized) > 50:
            raise ValueError("A maximum of 50 instruments is supported for correlation_matrix")

        min_samples_value = max(1, min(min_samples, 10_000))
        max_pairs = max(1, len(normalized) * len(normalized))
        limit_value = _clamp_limit(limit, max_pairs, max_pairs)

        filters = {
            "instruments": normalized,
            "date": date,
            "min_samples": min_samples_value,
            "limit": limit_value,
        }

        timer = time.perf_counter()
        matrix_payload = _fetch_correlation_matrix(info.context["ch_client"], filters)
        elapsed = time.perf_counter() - timer

        if not matrix_payload.get("date") or not matrix_payload.get("pairs"):
            logger.info(
                "analytics.correlation_matrix instruments=%s rows=0 elapsed=%.3fs",
                ",".join(normalized),
                elapsed,
            )
            return None

        instrument_order = normalized
        coefficients: Dict[str, Dict[str, Optional[float]]] = {
            inst: {other: None for other in instrument_order}
            for inst in instrument_order
        }

        for entry in matrix_payload.get("pairs", []):
            i1 = entry["instrument1"]
            i2 = entry["instrument2"]
            value = entry["correlation"]
            if i1 in coefficients and i2 in coefficients[i1]:
                coefficients[i1][i2] = value
            if i2 in coefficients and i1 in coefficients[i2]:
                coefficients[i2][i1] = value

        for inst in instrument_order:
            coefficients[inst][inst] = coefficients[inst].get(inst) or 1.0

        matrix = CorrelationMatrix(
            date=_date_from_iso(matrix_payload["date"]),
            instruments=instrument_order,
            coefficients=coefficients,
        )

        logger.info(
            "analytics.correlation_matrix instruments=%s rows=%d elapsed=%.3fs",
            ",".join(normalized),
            len(matrix_payload.get("pairs", [])),
            elapsed,
        )
        return matrix

    @strawberry.field
    def volatility_surface(
        self,
        info: Info,
        instrument_id: Optional[str] = None,
        instruments: Optional[List[str]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 1000,
    ) -> List[VolatilityPoint]:
        combined: List[str] = list(instruments or [])
        if instrument_id:
            combined.append(instrument_id)
        normalized = _normalize_instruments(combined)
        if not normalized:
            raise ValueError("Provide at least one instrument via instrument_id or instruments")

        limit_value = _clamp_limit(limit, 1000, 2_000)
        start, end = _ensure_date_range(
            start,
            end,
            _DEFAULT_VOL_WINDOW_DAYS,
            _MAX_VOL_WINDOW_DAYS,
        )

        filters = {
            "instruments": normalized,
            "start": start,
            "end": end,
            "limit": limit_value,
        }

        client: Client = info.context["ch_client"]
        timer = time.perf_counter()
        cached_rows = _fetch_volatility_surface(client, filters)
        elapsed = time.perf_counter() - timer

        points = [
            VolatilityPoint(
                as_of_date=_date_from_iso(row["as_of_date"]),
                instrument_id=row["instrument_id"],
                vol_30d=row.get("vol_30d"),
                vol_90d=row.get("vol_90d"),
                vol_365d=row.get("vol_365d"),
            )
            for row in cached_rows
            if row.get("as_of_date")
        ]

        logger.info(
            "analytics.volatility_surface instruments=%s rows=%d elapsed=%.3fs",
            ",".join(normalized),
            len(points),
            elapsed,
        )
        return points

    @strawberry.field
    def seasonality_decomposition(
        self,
        info: Info,
        instrument_id: str,
        method: str = "stl",
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 365,
    ) -> List[DecompositionPoint]:
        if not instrument_id:
            raise ValueError("instrument_id is required")
        normalized_instrument = instrument_id.strip().upper()
        method_normalized = method.strip().lower() or "stl"

        limit_value = _clamp_limit(limit, 365, 730)
        start, end = _ensure_date_range(
            start,
            end,
            _DEFAULT_SEASONALITY_WINDOW_DAYS,
            _MAX_SEASONALITY_WINDOW_DAYS,
        )

        filters = {
            "instrument_id": normalized_instrument,
            "method": method_normalized,
            "start": start,
            "end": end,
            "limit": limit_value,
        }

        client: Client = info.context["ch_client"]
        timer = time.perf_counter()
        cached_rows = _fetch_seasonality_points(client, filters)
        elapsed = time.perf_counter() - timer

        points = [
            DecompositionPoint(
                snapshot_date=_date_from_iso(row["snapshot_date"]),
                instrument_id=row["instrument_id"],
                method=row["method"],
                trend=row.get("trend"),
                seasonal=row.get("seasonal"),
                residual=row.get("residual"),
            )
            for row in cached_rows
            if row.get("snapshot_date")
        ]

        logger.info(
            "analytics.seasonality_decomposition instrument=%s method=%s rows=%d elapsed=%.3fs",
            normalized_instrument,
            method_normalized,
            len(points),
            elapsed,
        )
        return points

    @strawberry.field
    def seasonality_decomposition_latest(
        self,
        info: Info,
        instrument_id: str,
        method: str = "stl",
    ) -> Optional[DecompositionPoint]:
        if not instrument_id:
            raise ValueError("instrument_id is required")
        normalized_instrument = instrument_id.strip().upper()
        method_normalized = method.strip().lower() or "stl"

        filters = {
            "instrument_id": normalized_instrument,
            "method": method_normalized,
        }

        client: Client = info.context["ch_client"]
        timer = time.perf_counter()
        latest = _fetch_seasonality_latest(client, filters)
        elapsed = time.perf_counter() - timer

        if not latest:
            logger.info(
                "analytics.seasonality_decomposition_latest instrument=%s method=%s rows=0 elapsed=%.3fs",
                normalized_instrument,
                method_normalized,
                elapsed,
            )
            return None

        point = DecompositionPoint(
            snapshot_date=_date_from_iso(latest["snapshot_date"]),
            instrument_id=latest["instrument_id"],
            method=latest["method"],
            trend=latest.get("trend"),
            seasonal=latest.get("seasonal"),
            residual=latest.get("residual"),
        )

        logger.info(
            "analytics.seasonality_decomposition_latest instrument=%s method=%s rows=1 elapsed=%.3fs",
            normalized_instrument,
            method_normalized,
            elapsed,
        )
        return point

    @strawberry.field
    def research_query(
        self,
        info: Info,
        input: ResearchQueryInput,
    ) -> List[ResearchRow]:
        template = _RESEARCH_TEMPLATES.get(input.query_id)
        if not template:
            raise ValueError(f"Unsupported research query_id '{input.query_id}'")

        params = input.params or {}
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")

        limit_value = _clamp_limit(input.limit, template.default_limit, template.max_limit)
        sql, query_params, key_params = template.builder(params, limit_value)
        query_params["limit"] = limit_value
        key_params["limit"] = limit_value

        client: Client = info.context["ch_client"]
        timer = time.perf_counter()
        payload = _run_research_query_cached(
            client,
            input.query_id.value,
            sql,
            query_params,
            key_params,
        )
        elapsed = time.perf_counter() - timer

        rows = [
            ResearchRow(columns=payload["columns"], values=row)
            for row in payload.get("rows", [])
        ]

        logger.info(
            "analytics.research_query id=%s rows=%d elapsed=%.3fs",
            input.query_id.value,
            len(rows),
            elapsed,
        )
        return rows

    @strawberry.field
    def list_research_queries(self) -> List[str]:
        return [
            _RESEARCH_TEMPLATES[query_id].describe()
            for query_id in sorted(_RESEARCH_TEMPLATES.keys(), key=lambda q: q.value)
        ]

    @strawberry.field
    def portfolio_optimization_runs(
        self,
        info: Info,
        portfolio_id: Optional[str] = None,
        run_id: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 50,
    ) -> List[PortfolioOptimizationRun]:
        client: Client = info.context["ch_client"]
        conditions = ["1 = 1"]
        params: Dict[str, Any] = {"limit": limit}
        if portfolio_id:
            conditions.append("portfolio_id = %(portfolio_id)s")
            params["portfolio_id"] = portfolio_id
        if run_id:
            conditions.append("run_id = %(run_id)s")
            params["run_id"] = run_id
        if start:
            conditions.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            conditions.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                run_id,
                portfolio_id,
                as_of_date,
                method,
                params,
                weights,
                metrics,
                created_at,
                updated_at
            FROM ch.portfolio_optimization_runs
            WHERE {' AND '.join(conditions)}
            ORDER BY updated_at DESC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        outputs: List[PortfolioOptimizationRun] = []
        for row in rows:
            outputs.append(
                PortfolioOptimizationRun(
                    run_id=str(row["run_id"]),
                    portfolio_id=row["portfolio_id"],
                    as_of_date=row["as_of_date"],
                    method=row["method"],
                    params=_maybe_json(row.get("params")),
                    weights=_maybe_json(row.get("weights")) or {},
                    metrics=_maybe_json(row.get("metrics")) or {},
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at"),
                )
            )
        return outputs

    @strawberry.field
    def portfolio_risk_metrics(
        self,
        info: Info,
        portfolio_id: Optional[str] = None,
        run_id: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 100,
    ) -> List[PortfolioRiskMetric]:
        client: Client = info.context["ch_client"]
        conditions = ["1 = 1"]
        params: Dict[str, Any] = {"limit": limit}
        if portfolio_id:
            conditions.append("portfolio_id = %(portfolio_id)s")
            params["portfolio_id"] = portfolio_id
        if run_id:
            conditions.append("run_id = %(run_id)s")
            params["run_id"] = run_id
        if start:
            conditions.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            conditions.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                portfolio_id,
                run_id,
                volatility,
                variance,
                var_95,
                cvar_95,
                max_drawdown,
                beta,
                diversification_benefit,
                exposures,
                created_at
            FROM ch.portfolio_risk_metrics
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        outputs: List[PortfolioRiskMetric] = []
        for row in rows:
            outputs.append(
                PortfolioRiskMetric(
                    as_of_date=row["as_of_date"],
                    portfolio_id=row["portfolio_id"],
                    run_id=str(row["run_id"]),
                    volatility=row["volatility"],
                    variance=row["variance"],
                    var_95=row["var_95"],
                    cvar_95=row["cvar_95"],
                    max_drawdown=row["max_drawdown"],
                    beta=row.get("beta"),
                    diversification_benefit=row["diversification_benefit"],
                    exposures=_maybe_json(row.get("exposures")) or {},
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def portfolio_stress_results(
        self,
        info: Info,
        portfolio_id: Optional[str] = None,
        run_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 100,
    ) -> List[PortfolioStressResult]:
        client: Client = info.context["ch_client"]
        conditions = ["1 = 1"]
        params: Dict[str, Any] = {"limit": limit}
        if portfolio_id:
            conditions.append("portfolio_id = %(portfolio_id)s")
            params["portfolio_id"] = portfolio_id
        if run_id:
            conditions.append("run_id = %(run_id)s")
            params["run_id"] = run_id
        if scenario_id:
            conditions.append("scenario_id = %(scenario_id)s")
            params["scenario_id"] = scenario_id
        if start:
            conditions.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            conditions.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                portfolio_id,
                run_id,
                scenario_id,
                scenario,
                metrics,
                severity,
                probability,
                created_at
            FROM ch.portfolio_stress_results
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        outputs: List[PortfolioStressResult] = []
        for row in rows:
            outputs.append(
                PortfolioStressResult(
                    as_of_date=row["as_of_date"],
                    portfolio_id=row["portfolio_id"],
                    run_id=str(row["run_id"]),
                    scenario_id=row["scenario_id"],
                    scenario=_maybe_json(row.get("scenario")) or {},
                    metrics=_maybe_json(row.get("metrics")) or {},
                    severity=row["severity"],
                    probability=row["probability"],
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def arbitrage_opportunities(
        self,
        info: Info,
        commodities: Optional[List[str]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        min_confidence: float = 0.0,
        limit: int = 200,
    ) -> List[ArbitrageOpportunity]:
        client: Client = info.context["ch_client"]
        conditions = ["confidence >= %(min_confidence)s"]
        params: Dict[str, Any] = {"min_confidence": min_confidence, "limit": limit}

        if start:
            conditions.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            conditions.append("as_of_date <= %(end)s")
            params["end"] = end
        if commodities:
            normalized = sorted({commodity.upper() for commodity in commodities if commodity})
            if normalized:
                params["commodities"] = tuple(normalized)
                conditions.append("(commodity1 IN %(commodities)s OR commodity2 IN %(commodities)s)")

        query = f"""
            SELECT
                as_of_date,
                commodity1,
                commodity2,
                instrument1,
                instrument2,
                mean_spread,
                spread_volatility,
                transport_cost,
                storage_cost,
                net_profit,
                direction,
                confidence,
                period,
                metadata,
                created_at
            FROM ch.cross_market_arbitrage
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        outputs: List[ArbitrageOpportunity] = []
        for row in rows:
            outputs.append(
                ArbitrageOpportunity(
                    as_of_date=row["as_of_date"],
                    commodity1=row["commodity1"],
                    commodity2=row["commodity2"],
                    instrument1=row["instrument1"],
                    instrument2=row["instrument2"],
                    mean_spread=row["mean_spread"],
                    spread_volatility=row["spread_volatility"],
                    transport_cost=row["transport_cost"],
                    storage_cost=row["storage_cost"],
                    net_profit=row["net_profit"],
                    direction=row["direction"],
                    confidence=row["confidence"],
                    period=row.get("period"),
                    metadata=_maybe_json(row.get("metadata")),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def storage_arbitrage(
        self,
        info: Info,
        hub: str,
        as_of_date: date,
    ) -> Optional[StorageArbitrageOutput]:
        client: Client = info.context["ch_client"]
        query = """
            SELECT
                as_of_date,
                hub,
                region,
                expected_storage_value,
                breakeven_spread,
                optimal_schedule,
                cost_parameters,
                constraint_summary,
                diagnostics
            FROM ch.gas_storage_arbitrage
            WHERE hub = %(hub)s
              AND as_of_date = %(as_of)s
            ORDER BY created_at DESC
            LIMIT 1
        """
        result = _execute(client, query, {"hub": hub.upper(), "as_of": as_of_date})
        rows = _rows_to_dicts(result)
        if not rows:
            return None
        row = rows[0]
        schedule = [StorageScheduleEntry(**item) for item in json.loads(row["optimal_schedule"]) ]
        return StorageArbitrageOutput(
            as_of_date=row["as_of_date"],
            hub=row["hub"],
            region=row.get("region"),
            expected_storage_value=row["expected_storage_value"],
            breakeven_spread=row.get("breakeven_spread"),
            schedule=schedule,
            cost_parameters=json.loads(row["cost_parameters"]),
            constraint_summary=json.loads(row["constraint_summary"]),
            diagnostics=json.loads(row["diagnostics"]),
        )

    @strawberry.field
    def weather_impact(
        self,
        info: Info,
        entity_id: str,
        window: Optional[str] = None,
        limit: int = 5,
    ) -> List[WeatherImpactOutput]:
        client: Client = info.context["ch_client"]
        window_clause = ""
        params = {"entity": entity_id.upper(), "limit": limit}
        if window:
            window_clause = "AND window = %(window)s"
            params["window"] = window
        query = f"""
            SELECT
                date,
                entity_id,
                coef_type,
                coefficient,
                r2,
                window,
                diagnostics
            FROM ch.weather_impact
            WHERE entity_id = %(entity)s
              {window_clause}
            ORDER BY date DESC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        outputs: List[WeatherImpactOutput] = []
        for row in rows:
            diagnostics = json.loads(row["diagnostics"]) if row.get("diagnostics") else {}
            outputs.append(
                WeatherImpactOutput(
                    as_of_date=row["date"],
                    entity_id=row["entity_id"],
                    coef_type=row["coef_type"],
                    coefficient=row["coefficient"],
                    r2=row.get("r2"),
                    window=row["window"],
                    diagnostics=diagnostics,
                )
            )
        return outputs

    @strawberry.field
    def coal_to_gas_switching(
        self,
        info: Info,
        region: str,
        as_of_date: date,
    ) -> Optional[SwitchingEconomicsOutput]:
        client: Client = info.context["ch_client"]
        query = """
            SELECT
                as_of_date,
                region,
                coal_cost_mwh,
                gas_cost_mwh,
                co2_price,
                breakeven_gas_price,
                switch_share,
                diagnostics
            FROM ch.coal_gas_switching
            WHERE region = %(region)s
              AND as_of_date = %(as_of)s
            ORDER BY created_at DESC
            LIMIT 1
        """
        result = _execute(client, query, {"region": region.upper(), "as_of": as_of_date})
        rows = _rows_to_dicts(result)
        if not rows:
            return None
        row = rows[0]
        return SwitchingEconomicsOutput(
            as_of_date=row["as_of_date"],
            region=row["region"],
            coal_cost_mwh=row["coal_cost_mwh"],
            gas_cost_mwh=row["gas_cost_mwh"],
            co2_price=row["co2_price"],
            breakeven_gas_price=row["breakeven_gas_price"],
            switch_share=row["switch_share"],
            diagnostics=json.loads(row["diagnostics"]),
        )

    @strawberry.field
    def gas_basis_model(
        self,
        info: Info,
        hub: str,
        as_of_date: date,
    ) -> Optional[BasisModelOutput]:
        client: Client = info.context["ch_client"]
        query = """
            SELECT
                as_of_date,
                hub,
                predicted_basis,
                actual_basis,
                method,
                diagnostics,
                feature_snapshot
            FROM ch.gas_basis_models
            WHERE hub = %(hub)s
              AND as_of_date = %(as_of)s
            ORDER BY created_at DESC
            LIMIT 1
        """
        result = _execute(client, query, {"hub": hub.upper(), "as_of": as_of_date})
        rows = _rows_to_dicts(result)
        if not rows:
            return None
        row = rows[0]
        return BasisModelOutput(
            as_of_date=row["as_of_date"],
            hub=row["hub"],
            predicted_basis=row["predicted_basis"],
            actual_basis=row.get("actual_basis"),
            method=row["method"],
            diagnostics=json.loads(row["diagnostics"]),
            feature_snapshot=json.loads(row["feature_snapshot"]),
        )

    @strawberry.field
    def lng_routing(
        self,
        info: Info,
        as_of: date,
        export_terminals: Optional[List[str]] = None,
        import_terminals: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Optional[LNGRoutingResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"as_of": as_of, "limit": limit}
        clauses = ["as_of_date = %(as_of)s"]
        if export_terminals:
            params["exports"] = tuple(export_terminals)
            clauses.append("export_terminal IN %(exports)s")
        if import_terminals:
            params["imports"] = tuple(import_terminals)
            clauses.append("import_terminal IN %(imports)s")

        where_clause = " AND ".join(clauses)
        query = f"""
            SELECT
                as_of_date,
                route_id,
                export_terminal,
                import_terminal,
                vessel_type,
                cargo_size_bcf,
                vessel_speed_knots,
                fuel_price_usd_per_tonne,
                distance_nm,
                voyage_time_days,
                fuel_consumption_tonnes,
                fuel_cost_usd,
                charter_cost_usd,
                port_cost_usd,
                total_cost_usd,
                cost_per_mmbtu_usd,
                is_optimal_route,
                assumptions,
                diagnostics,
                model_version
            FROM ch.lng_routing_optimization
            WHERE {where_clause}
            ORDER BY is_optimal_route DESC, total_cost_usd ASC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        if not rows:
            return None

        options: List[LNGRoutingOption] = []
        for row in rows:
            options.append(
                LNGRoutingOption(
                    route_id=row["route_id"],
                    export_terminal=row["export_terminal"],
                    import_terminal=row["import_terminal"],
                    vessel_type=row.get("vessel_type"),
                    cargo_size_bcf=row["cargo_size_bcf"],
                    vessel_speed_knots=row["vessel_speed_knots"],
                    fuel_price_usd_per_tonne=row["fuel_price_usd_per_tonne"],
                    distance_nm=row["distance_nm"],
                    voyage_time_days=row["voyage_time_days"],
                    fuel_consumption_tonnes=row["fuel_consumption_tonnes"],
                    fuel_cost_usd=row["fuel_cost_usd"],
                    charter_cost_usd=row["charter_cost_usd"],
                    port_cost_usd=row.get("port_cost_usd"),
                    total_cost_usd=row["total_cost_usd"],
                    cost_per_mmbtu_usd=row["cost_per_mmbtu_usd"],
                    is_optimal_route=bool(row["is_optimal_route"]),
                    assumptions=_maybe_json(row.get("assumptions")),
                    diagnostics=_maybe_json(row.get("diagnostics")),
                    model_version=row.get("model_version"),
                )
            )

        metadata = {
            "export_terminals": export_terminals or "ALL",
            "import_terminals": import_terminals or "ALL",
            "result_count": len(options),
        }
        return LNGRoutingResult(as_of_date=as_of, options=options, metadata=metadata)

    @strawberry.field
    def coal_transport_costs(
        self,
        info: Info,
        route_id: str,
        month: Optional[date] = None,
        transport_mode: Optional[str] = None,
        limit: int = 24,
    ) -> List[CoalTransportCost]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"route": route_id, "limit": limit}
        clauses = ["route_id = %(route)s"]
        if month:
            clauses.append("as_of_month = %(month)s")
            params["month"] = month
        if transport_mode:
            clauses.append("transport_mode = %(mode)s")
            params["mode"] = transport_mode

        where_clause = " AND ".join(clauses)
        query = f"""
            SELECT
                as_of_month,
                route_id,
                origin_region,
                destination_region,
                transport_mode,
                vessel_type,
                cargo_tonnes,
                fuel_price_usd_per_tonne,
                freight_cost_usd,
                bunker_cost_usd,
                port_fees_usd,
                congestion_premium_usd,
                carbon_cost_usd,
                demurrage_cost_usd,
                rail_cost_usd,
                truck_cost_usd,
                total_cost_usd,
                currency,
                assumptions,
                diagnostics,
                model_version
            FROM ch.coal_transport_costs
            WHERE {where_clause}
            ORDER BY as_of_month DESC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        outputs: List[CoalTransportCost] = []
        for row in rows:
            breakdown = CoalRouteBreakdown(
                bunker_cost_usd=row.get("bunker_cost_usd"),
                port_fees_usd=row.get("port_fees_usd"),
                congestion_premium_usd=row.get("congestion_premium_usd"),
                carbon_cost_usd=row.get("carbon_cost_usd"),
                demurrage_cost_usd=row.get("demurrage_cost_usd"),
                rail_cost_usd=row.get("rail_cost_usd"),
                truck_cost_usd=row.get("truck_cost_usd"),
            )
            outputs.append(
                CoalTransportCost(
                    as_of_month=row["as_of_month"],
                    route_id=row["route_id"],
                    origin_region=row["origin_region"],
                    destination_region=row["destination_region"],
                    transport_mode=row["transport_mode"],
                    vessel_type=row.get("vessel_type"),
                    cargo_tonnes=row["cargo_tonnes"],
                    fuel_price_usd_per_tonne=row["fuel_price_usd_per_tonne"],
                    freight_cost_usd=row["freight_cost_usd"],
                    total_cost_usd=row["total_cost_usd"],
                    currency=row["currency"],
                    breakdown=breakdown,
                    assumptions=_maybe_json(row.get("assumptions")),
                    diagnostics=_maybe_json(row.get("diagnostics")),
                    model_version=row.get("model_version"),
                )
            )
        return outputs

    @strawberry.field
    def pipeline_congestion(
        self,
        info: Info,
        pipeline_id: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 90,
    ) -> List[PipelineCongestionPoint]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"pipeline": pipeline_id, "limit": limit}
        clauses = ["pipeline_id = %(pipeline)s"]
        if start:
            clauses.append("forecast_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("forecast_date <= %(end)s")
            params["end"] = end

        where_clause = " AND ".join(clauses)
        query = f"""
            SELECT
                forecast_date,
                utilization_forecast_pct,
                congestion_probability,
                risk_score,
                risk_tier,
                alert_level,
                drivers,
                diagnostics,
                model_version
            FROM ch.pipeline_congestion_forecast
            WHERE {where_clause}
            ORDER BY forecast_date DESC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        outputs: List[PipelineCongestionPoint] = []
        for row in rows:
            outputs.append(
                PipelineCongestionPoint(
                    forecast_date=row["forecast_date"],
                    utilization_forecast_pct=row["utilization_forecast_pct"],
                    congestion_probability=row["congestion_probability"],
                    risk_score=row["risk_score"],
                    risk_tier=row["risk_tier"],
                    alert_level=row.get("alert_level"),
                    drivers=_maybe_json(row.get("drivers")),
                    diagnostics=_maybe_json(row.get("diagnostics")),
                    model_version=row.get("model_version"),
                )
            )
        return outputs

    @strawberry.field
    def pipeline_alerts(
        self,
        info: Info,
        pipeline_id: str,
        lookahead_days: int = 7,
    ) -> List[PipelineCongestionAlert]:
        client: Client = info.context["ch_client"]
        query = """
            SELECT
                forecast_date,
                utilization_forecast_pct,
                risk_tier,
                alert_level
            FROM ch.pipeline_congestion_forecast
            WHERE pipeline_id = %(pipeline)s
              AND forecast_date BETWEEN today() AND today() + toIntervalDay(%(lookahead)s)
              AND risk_tier IN ('high', 'critical')
            ORDER BY forecast_date ASC
        """
        rows = _rows_to_dicts(_execute(client, query, {"pipeline": pipeline_id, "lookahead": lookahead_days}))
        alerts: List[PipelineCongestionAlert] = []
        for row in rows:
            severity = row.get("risk_tier", "high").upper()
            message = f"{severity} congestion risk for {pipeline_id} on {row['forecast_date']}"
            alerts.append(
                PipelineCongestionAlert(
                    date=row["forecast_date"],
                    utilization_forecast_pct=row["utilization_forecast_pct"],
                    risk_tier=row["risk_tier"],
                    alert_level=row.get("alert_level"),
                    message=message,
                )
            )
        return alerts

    @strawberry.field
    def crack_optimization_results(
        self,
        info: Info,
        region: Optional[str] = None,
        crack_type: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[CrackOptimizationResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if region:
            clauses.append("region = %(region)s")
            params["region"] = region
        if crack_type:
            clauses.append("crack_type = %(crack_type)s")
            params["crack_type"] = crack_type
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                region,
                refinery_id,
                crack_type,
                crude_code,
                gasoline_price,
                diesel_price,
                jet_price,
                crack_spread,
                margin_per_bbl,
                optimal_yields,
                constraints,
                diagnostics,
                model_version,
                created_at
            FROM ch.refining_crack_optimization
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[CrackOptimizationResult] = []
        for row in rows:
            outputs.append(
                CrackOptimizationResult(
                    as_of_date=row["as_of_date"],
                    region=row["region"],
                    refinery_id=row.get("refinery_id"),
                    crack_type=row["crack_type"],
                    crude_code=row["crude_code"],
                    gasoline_price=row["gasoline_price"],
                    diesel_price=row["diesel_price"],
                    jet_price=row.get("jet_price"),
                    crack_spread=row["crack_spread"],
                    margin_per_bbl=row["margin_per_bbl"],
                    optimal_yields=_maybe_json(row.get("optimal_yields")),
                    constraints=_maybe_json(row.get("constraints")),
                    diagnostics=_maybe_json(row.get("diagnostics")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def refinery_yield_results(
        self,
        info: Info,
        crude_type: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[RefineryYieldResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if crude_type:
            clauses.append("crude_type = %(crude_type)s")
            params["crude_type"] = crude_type
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                crude_type,
                process_config,
                yields,
                value_per_bbl,
                operating_cost,
                net_value,
                diagnostics,
                model_version,
                created_at
            FROM ch.refinery_yield_model
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[RefineryYieldResult] = []
        for row in rows:
            outputs.append(
                RefineryYieldResult(
                    as_of_date=row["as_of_date"],
                    crude_type=row["crude_type"],
                    process_config=_maybe_json(row.get("process_config")),
                    yields=_maybe_json(row.get("yields")),
                    value_per_bbl=row["value_per_bbl"],
                    operating_cost=row["operating_cost"],
                    net_value=row["net_value"],
                    diagnostics=_maybe_json(row.get("diagnostics")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def product_elasticities(
        self,
        info: Info,
        product: Optional[str] = None,
        region: Optional[str] = None,
        method: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[ProductElasticityResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if product:
            clauses.append("product = %(product)s")
            params["product"] = product
        if region:
            clauses.append("region = %(region)s")
            params["region"] = region
        if method:
            clauses.append("method = %(method)s")
            params["method"] = method
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                product,
                region,
                method,
                elasticity,
                r_squared,
                own_or_cross,
                product_pair,
                data_points,
                diagnostics,
                model_version,
                created_at
            FROM ch.product_demand_elasticity
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[ProductElasticityResult] = []
        for row in rows:
            outputs.append(
                ProductElasticityResult(
                    as_of_date=row["as_of_date"],
                    product=row["product"],
                    region=row.get("region"),
                    method=row["method"],
                    elasticity=row["elasticity"],
                    r_squared=row.get("r_squared"),
                    own_or_cross=row["own_or_cross"],
                    product_pair=row.get("product_pair"),
                    data_points=row["data_points"],
                    diagnostics=_maybe_json(row.get("diagnostics")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def transport_fuel_substitution_metrics(
        self,
        info: Info,
        region: Optional[str] = None,
        metric: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[TransportFuelSubstitutionMetric]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if region:
            clauses.append("region = %(region)s")
            params["region"] = region
        if metric:
            clauses.append("metric = %(metric)s")
            params["metric"] = metric
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                region,
                metric,
                value,
                details,
                model_version,
                created_at
            FROM ch.transport_fuel_substitution
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[TransportFuelSubstitutionMetric] = []
        for row in rows:
            outputs.append(
                TransportFuelSubstitutionMetric(
                    as_of_date=row["as_of_date"],
                    region=row["region"],
                    metric=row["metric"],
                    value=row["value"],
                    details=_maybe_json(row.get("details")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def rin_price_forecasts(
        self,
        info: Info,
        rin_category: Optional[str] = None,
        as_of_date: Optional[date] = None,
        forecast_date: Optional[date] = None,
        horizon_days: Optional[int] = None,
        limit: int = 365,
    ) -> List[RINPriceForecastPoint]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if rin_category:
            clauses.append("rin_category = %(category)s")
            params["category"] = rin_category
        if as_of_date:
            clauses.append("as_of_date = %(as_of)s")
            params["as_of"] = as_of_date
        if forecast_date:
            clauses.append("forecast_date = %(forecast)s")
            params["forecast"] = forecast_date
        if horizon_days is not None:
            clauses.append("horizon_days = %(horizon)s")
            params["horizon"] = horizon_days

        query = f"""
            SELECT
                as_of_date,
                rin_category,
                horizon_days,
                forecast_date,
                forecast_price,
                std,
                drivers,
                model_version,
                created_at
            FROM ch.rin_price_forecast
            WHERE {' AND '.join(clauses)}
            ORDER BY forecast_date ASC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[RINPriceForecastPoint] = []
        for row in rows:
            outputs.append(
                RINPriceForecastPoint(
                    as_of_date=row["as_of_date"],
                    rin_category=row["rin_category"],
                    horizon_days=row["horizon_days"],
                    forecast_date=row["forecast_date"],
                    forecast_price=row["forecast_price"],
                    std=row.get("std"),
                    drivers=_maybe_json(row.get("drivers")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def carbon_price_forecasts(
        self,
        info: Info,
        market: Optional[str] = None,
        as_of_date: Optional[date] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        horizon_days: Optional[int] = None,
        limit: int = 365,
    ) -> List[CarbonPriceForecastPoint]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if market:
            clauses.append("market = %(market)s")
            params["market"] = market
        if as_of_date:
            clauses.append("as_of_date = %(as_of)s")
            params["as_of"] = as_of_date
        if start:
            clauses.append("forecast_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("forecast_date <= %(end)s")
            params["end"] = end
        if horizon_days is not None:
            clauses.append("horizon_days = %(horizon)s")
            params["horizon"] = horizon_days

        query = f"""
            SELECT
                as_of_date,
                market,
                horizon_days,
                forecast_date,
                forecast_price,
                std,
                drivers,
                model_version,
                created_at
            FROM ch.carbon_price_forecast
            WHERE {' AND '.join(clauses)}
            ORDER BY forecast_date ASC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[CarbonPriceForecastPoint] = []
        for row in rows:
            outputs.append(
                CarbonPriceForecastPoint(
                    as_of_date=row["as_of_date"],
                    market=row["market"],
                    horizon_days=row["horizon_days"],
                    forecast_date=row["forecast_date"],
                    forecast_price=row["forecast_price"],
                    std=row.get("std"),
                    drivers=_maybe_json(row.get("drivers")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def compliance_costs(
        self,
        info: Info,
        market: Optional[str] = None,
        sector: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[ComplianceCostResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if market:
            clauses.append("market = %(market)s")
            params["market"] = market
        if sector:
            clauses.append("sector = %(sector)s")
            params["sector"] = sector
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                market,
                coalesce(sector, 'aggregate') AS sector,
                total_emissions,
                average_price,
                cost_per_tonne,
                total_compliance_cost,
                details,
                model_version,
                created_at
            FROM ch.carbon_compliance_costs
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[ComplianceCostResult] = []
        for row in rows:
            outputs.append(
                ComplianceCostResult(
                    as_of_date=row["as_of_date"],
                    market=row["market"],
                    sector=row["sector"],
                    total_emissions=row["total_emissions"],
                    average_price=row["average_price"],
                    cost_per_tonne=row["cost_per_tonne"],
                    total_compliance_cost=row["total_compliance_cost"],
                    details=_maybe_json(row.get("details")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def carbon_leakage(
        self,
        info: Info,
        sector: Optional[str] = None,
        risk_level: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[CarbonLeakageRisk]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if sector:
            clauses.append("sector = %(sector)s")
            params["sector"] = sector
        if risk_level:
            clauses.append("risk_level = %(risk)s")
            params["risk"] = risk_level
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                sector,
                domestic_price,
                international_price,
                price_differential,
                trade_exposure,
                emissions_intensity,
                leakage_risk_score,
                risk_level,
                details,
                model_version,
                created_at
            FROM ch.carbon_leakage_risk
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[CarbonLeakageRisk] = []
        for row in rows:
            outputs.append(
                CarbonLeakageRisk(
                    as_of_date=row["as_of_date"],
                    sector=row["sector"],
                    domestic_price=row["domestic_price"],
                    international_price=row["international_price"],
                    price_differential=row["price_differential"],
                    trade_exposure=row["trade_exposure"],
                    emissions_intensity=row["emissions_intensity"],
                    leakage_risk_score=row["leakage_risk_score"],
                    risk_level=row["risk_level"],
                    details=_maybe_json(row.get("details")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def decarbonization_pathways(
        self,
        info: Info,
        sector: Optional[str] = None,
        scenario: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[DecarbonizationPathwayResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if sector:
            clauses.append("sector = %(sector)s")
            params["sector"] = sector
        if scenario:
            clauses.append("policy_scenario = %(scenario)s")
            params["scenario"] = scenario
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                sector,
                policy_scenario,
                target_year,
                annual_reduction_rate,
                cumulative_emissions,
                target_achieved,
                emissions_trajectory,
                technology_analysis,
                model_version,
                created_at
            FROM ch.decarbonization_pathways
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[DecarbonizationPathwayResult] = []
        for row in rows:
            outputs.append(
                DecarbonizationPathwayResult(
                    as_of_date=row["as_of_date"],
                    sector=row["sector"],
                    policy_scenario=row["policy_scenario"],
                    target_year=row["target_year"],
                    annual_reduction_rate=row["annual_reduction_rate"],
                    cumulative_emissions=row["cumulative_emissions"],
                    target_achieved=bool(row.get("target_achieved", 0)),
                    emissions_trajectory=_maybe_json(row.get("emissions_trajectory")),
                    technology_analysis=_maybe_json(row.get("technology_analysis")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def renewable_adoption(
        self,
        info: Info,
        technology: Optional[str] = None,
        as_of_date: Optional[date] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[RenewableAdoptionPoint]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if technology:
            clauses.append("technology = %(technology)s")
            params["technology"] = technology
        if as_of_date:
            clauses.append("as_of_date = %(as_of)s")
            params["as_of"] = as_of_date
        if start:
            clauses.append("forecast_year >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("forecast_year <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                technology,
                forecast_year,
                capacity_gw,
                policy_support,
                economic_multipliers,
                assumptions,
                model_version,
                created_at
            FROM ch.renewable_adoption_forecast
            WHERE {' AND '.join(clauses)}
            ORDER BY forecast_year ASC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[RenewableAdoptionPoint] = []
        for row in rows:
            outputs.append(
                RenewableAdoptionPoint(
                    as_of_date=row["as_of_date"],
                    technology=row["technology"],
                    forecast_year=row["forecast_year"],
                    capacity_gw=row["capacity_gw"],
                    policy_support=row["policy_support"],
                    economic_multipliers=_maybe_json(row.get("economic_multipliers")),
                    assumptions=_maybe_json(row.get("assumptions")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def stranded_asset_risk(
        self,
        info: Info,
        asset_type: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[StrandedAssetRisk]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if asset_type:
            clauses.append("asset_type = %(asset_type)s")
            params["asset_type"] = asset_type
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                asset_type,
                asset_value,
                carbon_cost_pv,
                stranded_value,
                stranded_ratio,
                remaining_lifetime,
                risk_level,
                details,
                model_version,
                created_at
            FROM ch.stranded_asset_risk
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[StrandedAssetRisk] = []
        for row in rows:
            outputs.append(
                StrandedAssetRisk(
                    as_of_date=row["as_of_date"],
                    asset_type=row["asset_type"],
                    asset_value=row["asset_value"],
                    carbon_cost_pv=row["carbon_cost_pv"],
                    stranded_value=row["stranded_value"],
                    stranded_ratio=row["stranded_ratio"],
                    remaining_lifetime=row["remaining_lifetime"],
                    risk_level=row["risk_level"],
                    details=_maybe_json(row.get("details")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def policy_scenario_impacts(
        self,
        info: Info,
        scenario: Optional[str] = None,
        entity: Optional[str] = None,
        metric: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 500,
    ) -> List[PolicyScenarioImpact]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if scenario:
            clauses.append("scenario = %(scenario)s")
            params["scenario"] = scenario
        if entity:
            clauses.append("entity = %(entity)s")
            params["entity"] = entity
        if metric:
            clauses.append("metric = %(metric)s")
            params["metric"] = metric
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                scenario,
                entity,
                metric,
                value,
                details,
                model_version,
                created_at
            FROM ch.policy_scenario_impacts
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC, scenario, entity
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[PolicyScenarioImpact] = []
        for row in rows:
            outputs.append(
                PolicyScenarioImpact(
                    as_of_date=row["as_of_date"],
                    scenario=row["scenario"],
                    entity=row["entity"],
                    metric=row["metric"],
                    value=row["value"],
                    details=_maybe_json(row.get("details")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def biodiesel_spread_results(
        self,
        info: Info,
        region: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[BiodieselSpreadResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if region:
            clauses.append("region = %(region)s")
            params["region"] = region
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                region,
                mean_gross_spread,
                mean_net_spread,
                spread_volatility,
                arbitrage_opportunities,
                diagnostics,
                model_version,
                created_at
            FROM ch.biodiesel_diesel_spread
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[BiodieselSpreadResult] = []
        for row in rows:
            outputs.append(
                BiodieselSpreadResult(
                    as_of_date=row["as_of_date"],
                    region=row.get("region"),
                    mean_gross_spread=row["mean_gross_spread"],
                    mean_net_spread=row["mean_net_spread"],
                    spread_volatility=row["spread_volatility"],
                    arbitrage_opportunities=row["arbitrage_opportunities"],
                    diagnostics=_maybe_json(row.get("diagnostics")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def carbon_intensity_results(
        self,
        info: Info,
        fuel_type: Optional[str] = None,
        pathway: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[CarbonIntensityResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if fuel_type:
            clauses.append("fuel_type = %(fuel_type)s")
            params["fuel_type"] = fuel_type
        if pathway:
            clauses.append("pathway = %(pathway)s")
            params["pathway"] = pathway
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                fuel_type,
                pathway,
                total_ci,
                base_emissions,
                transport_emissions,
                land_use_emissions,
                ci_per_mj,
                assumptions,
                model_version,
                created_at
            FROM ch.carbon_intensity_results
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[CarbonIntensityResult] = []
        for row in rows:
            outputs.append(
                CarbonIntensityResult(
                    as_of_date=row["as_of_date"],
                    fuel_type=row["fuel_type"],
                    pathway=row["pathway"],
                    total_ci=row["total_ci"],
                    base_emissions=row["base_emissions"],
                    transport_emissions=row["transport_emissions"],
                    land_use_emissions=row["land_use_emissions"],
                    ci_per_mj=row["ci_per_mj"],
                    assumptions=_maybe_json(row.get("assumptions")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def renewables_policy_impacts(
        self,
        info: Info,
        policy: Optional[str] = None,
        entity: Optional[str] = None,
        metric: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 200,
    ) -> List[RenewablesPolicyImpactResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"limit": limit}
        clauses = ["1 = 1"]
        if policy:
            clauses.append("policy = %(policy)s")
            params["policy"] = policy
        if entity:
            clauses.append("entity = %(entity)s")
            params["entity"] = entity
        if metric:
            clauses.append("metric = %(metric)s")
            params["metric"] = metric
        if start:
            clauses.append("as_of_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("as_of_date <= %(end)s")
            params["end"] = end

        query = f"""
            SELECT
                as_of_date,
                policy,
                entity,
                metric,
                value,
                details,
                model_version,
                created_at
            FROM ch.renewables_policy_impact
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date DESC
            LIMIT %(limit)s
        """
        rows = _rows_to_dicts(_execute(client, query, params))
        outputs: List[RenewablesPolicyImpactResult] = []
        for row in rows:
            outputs.append(
                RenewablesPolicyImpactResult(
                    as_of_date=row["as_of_date"],
                    policy=row["policy"],
                    entity=row["entity"],
                    metric=row["metric"],
                    value=row["value"],
                    details=_maybe_json(row.get("details")),
                    model_version=row.get("model_version"),
                    created_at=row.get("created_at"),
                )
            )
        return outputs

    @strawberry.field
    def seasonal_demand(
        self,
        info: Info,
        region: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        scenario_id: Optional[str] = None,
        limit: int = 180,
    ) -> Optional[SeasonalDemandResult]:
        client: Client = info.context["ch_client"]
        params: Dict[str, Any] = {"region": region, "limit": limit}
        clauses = ["region = %(region)s"]
        if start:
            clauses.append("forecast_date >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("forecast_date <= %(end)s")
            params["end"] = end
        if scenario_id:
            clauses.append("scenario_id = %(scenario)s")
            params["scenario"] = scenario_id

        where_clause = " AND ".join(clauses)
        query = f"""
            SELECT
                forecast_date,
                region,
                sector,
                scenario_id,
                base_demand_mw,
                weather_adjustment_mw,
                economic_adjustment_mw,
                holiday_adjustment_mw,
                final_forecast_mw,
                peak_risk_score,
                confidence_low_mw,
                confidence_high_mw,
                diagnostics,
                model_version
            FROM ch.seasonal_demand_forecast
            WHERE {where_clause}
            ORDER BY forecast_date ASC
            LIMIT %(limit)s
        """
        result = _execute(client, query, params)
        rows = _rows_to_dicts(result)
        if not rows:
            return None

        points: List[DemandForecastPoint] = []
        risk_values: List[float] = []
        for row in rows:
            risk = row.get("peak_risk_score")
            if risk is not None:
                risk_values.append(risk)
            points.append(
                DemandForecastPoint(
                    forecast_date=row["forecast_date"],
                    region=row["region"],
                    sector=row.get("sector"),
                    scenario_id=row["scenario_id"],
                    base_demand_mw=row["base_demand_mw"],
                    weather_adjustment_mw=row.get("weather_adjustment_mw"),
                    economic_adjustment_mw=row.get("economic_adjustment_mw"),
                    holiday_adjustment_mw=row.get("holiday_adjustment_mw"),
                    final_forecast_mw=row["final_forecast_mw"],
                    peak_risk_score=risk,
                    confidence_low_mw=row.get("confidence_low_mw"),
                    confidence_high_mw=row.get("confidence_high_mw"),
                    diagnostics=_maybe_json(row.get("diagnostics")),
                    model_version=row.get("model_version"),
                )
            )

        forecast_peak = max(point.final_forecast_mw for point in points)
        avg_risk = sum(risk_values) / len(risk_values) if risk_values else None
        observation_payload = {
            "count": len(points),
            "start": str(points[0].forecast_date),
            "end": str(points[-1].forecast_date),
        }

        assessment = PeakDemandAssessment(
            forecast_peak_mw=forecast_peak,
            average_peak_risk=avg_risk,
            observations=observation_payload,
        )

        scenario_value = scenario_id or points[0].scenario_id
        return SeasonalDemandResult(
            region=region,
            scenario_id=scenario_value,
            points=points,
            peak_assessment=assessment,
        )


__all__ = [
    "AnalyticsQuery",
    "CrackOptimizationResult",
    "RefineryYieldResult",
    "ProductElasticityResult",
    "TransportFuelSubstitutionMetric",
    "RINPriceForecastPoint",
    "BiodieselSpreadResult",
    "CarbonIntensityResult",
    "RenewablesPolicyImpactResult",
    "StorageArbitrageOutput",
    "WeatherImpactOutput",
    "SwitchingEconomicsOutput",
    "BasisModelOutput",
    "LNGRoutingResult",
    "LNGRoutingOption",
    "CoalTransportCost",
    "CoalRouteBreakdown",
    "PipelineCongestionPoint",
    "PipelineCongestionAlert",
    "SeasonalDemandResult",
    "DemandForecastPoint",
    "PeakDemandAssessment",
]
