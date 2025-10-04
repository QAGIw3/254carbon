"""GraphQL schema extensions for gas & coal analytics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import strawberry
from clickhouse_driver import Client
from strawberry.types import Info


@dataclass
class ClickHouseResult:
    data: list
    columns: list


def _execute(client: Client, query: str, parameters: dict) -> ClickHouseResult:
    data, column_types = client.execute(query, parameters, with_column_types=True)
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


class AnalyticsQuery:
    """Analytics fields for gas & coal workflows."""

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
