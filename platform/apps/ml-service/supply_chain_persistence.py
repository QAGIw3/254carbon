"""Persistence helpers for supply chain analytics outputs."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from clickhouse_driver import Client

logger = logging.getLogger(__name__)


def _json_or_none(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    try:
        return json.dumps(payload, default=str)
    except (TypeError, ValueError):
        logger.warning("Failed to serialize payload for persistence", exc_info=True)
        return None


class SupplyChainPersistence:
    """Handles inserts of supply chain analytics artefacts into ClickHouse."""

    def __init__(
        self,
        *,
        ch_client: Optional[Client] = None,
        host: str = "clickhouse",
        port: int = 9000,
        database: str = "ch",
    ) -> None:
        self._external_client = ch_client is not None
        self.client = ch_client or Client(host=host, port=port, database=database)

    # ------------------------------------------------------------------
    # LNG routing optimisation
    # ------------------------------------------------------------------
    def persist_lng_routing(
        self,
        *,
        as_of: date,
        routes: Sequence[Dict[str, Any]],
        model_version: str,
        assumptions: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not routes:
            return 0

        rows: List[Tuple[Any, ...]] = []
        assumption_payload = _json_or_none(assumptions)
        diagnostics_payload = _json_or_none(diagnostics)

        for route in routes:
            rows.append(
                (
                    as_of,
                    route.get("route_id"),
                    route.get("export_terminal"),
                    route.get("import_terminal"),
                    route.get("vessel_type"),
                    float(route.get("cargo_size_bcf", 0.0)),
                    float(route.get("vessel_speed_knots", 0.0)),
                    float(route.get("fuel_price_usd_per_tonne", 0.0)),
                    float(route.get("distance_nm", 0.0)),
                    float(route.get("voyage_time_days", 0.0)),
                    float(route.get("fuel_consumption_tonnes", 0.0)),
                    float(route.get("fuel_cost_usd", 0.0)),
                    float(route.get("charter_cost_usd", 0.0)),
                    route.get("port_cost_usd"),
                    float(route.get("total_cost_usd", 0.0)),
                    float(route.get("cost_per_mmbtu_usd", 0.0)),
                    int(route.get("is_optimal_route", 0)),
                    assumption_payload,
                    diagnostics_payload,
                    model_version,
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.lng_routing_optimization
                (as_of_date, route_id, export_terminal, import_terminal, vessel_type,
                 cargo_size_bcf, vessel_speed_knots, fuel_price_usd_per_tonne,
                 distance_nm, voyage_time_days, fuel_consumption_tonnes,
                 fuel_cost_usd, charter_cost_usd, port_cost_usd, total_cost_usd,
                 cost_per_mmbtu_usd, is_optimal_route, assumptions, diagnostics, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Coal transport costs
    # ------------------------------------------------------------------
    def persist_coal_transport_costs(
        self,
        *,
        as_of_month: date,
        records: Sequence[Dict[str, Any]],
        model_version: str,
        assumptions: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not records:
            return 0

        assumption_payload = _json_or_none(assumptions)
        diagnostics_payload = _json_or_none(diagnostics)
        rows: List[Tuple[Any, ...]] = []

        for record in records:
            rows.append(
                (
                    as_of_month,
                    record.get("route_id"),
                    record.get("origin_region"),
                    record.get("destination_region"),
                    record.get("transport_mode"),
                    record.get("vessel_type"),
                    float(record.get("cargo_tonnes", 0.0)),
                    float(record.get("fuel_price_usd_per_tonne", 0.0)),
                    float(record.get("freight_cost_usd", 0.0)),
                    record.get("bunker_cost_usd"),
                    record.get("port_fees_usd"),
                    record.get("congestion_premium_usd"),
                    record.get("carbon_cost_usd"),
                    record.get("demurrage_cost_usd"),
                    record.get("rail_cost_usd"),
                    record.get("truck_cost_usd"),
                    float(record.get("total_cost_usd", 0.0)),
                    record.get("currency", "USD"),
                    assumption_payload,
                    diagnostics_payload,
                    model_version,
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.coal_transport_costs
                (as_of_month, route_id, origin_region, destination_region, transport_mode,
                 vessel_type, cargo_tonnes, fuel_price_usd_per_tonne, freight_cost_usd,
                 bunker_cost_usd, port_fees_usd, congestion_premium_usd, carbon_cost_usd,
                 demurrage_cost_usd, rail_cost_usd, truck_cost_usd, total_cost_usd, currency,
                 assumptions, diagnostics, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Pipeline congestion forecasts
    # ------------------------------------------------------------------
    def persist_pipeline_forecast(
        self,
        *,
        forecasts: Sequence[Dict[str, Any]],
        model_version: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not forecasts:
            return 0

        diagnostics_payload = _json_or_none(diagnostics)
        rows: List[Tuple[Any, ...]] = []

        for forecast in forecasts:
            rows.append(
                (
                    forecast.get("forecast_date"),
                    forecast.get("pipeline_id"),
                    forecast.get("market"),
                    forecast.get("segment"),
                    int(forecast.get("horizon_days", 0)),
                    float(forecast.get("utilization_forecast_pct", 0.0)),
                    forecast.get("utilization_actual_pct"),
                    float(forecast.get("congestion_probability", 0.0)),
                    float(forecast.get("risk_score", 0.0)),
                    forecast.get("risk_tier"),
                    forecast.get("alert_level"),
                    _json_or_none(forecast.get("drivers")),
                    diagnostics_payload,
                    model_version,
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.pipeline_congestion_forecast
                (forecast_date, pipeline_id, market, segment, horizon_days,
                 utilization_forecast_pct, utilization_actual_pct,
                 congestion_probability, risk_score, risk_tier, alert_level,
                 drivers, diagnostics, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Seasonal demand forecasts
    # ------------------------------------------------------------------
    def persist_seasonal_demand(
        self,
        *,
        records: Sequence[Dict[str, Any]],
        model_version: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not records:
            return 0

        diagnostics_payload = _json_or_none(diagnostics)
        rows: List[Tuple[Any, ...]] = []

        for record in records:
            rows.append(
                (
                    record.get("forecast_date"),
                    record.get("region"),
                    record.get("sector"),
                    record.get("scenario_id", "BASE"),
                    float(record.get("base_demand_mw", 0.0)),
                    record.get("weather_adjustment_mw"),
                    record.get("economic_adjustment_mw"),
                    record.get("holiday_adjustment_mw"),
                    float(record.get("final_forecast_mw", 0.0)),
                    record.get("peak_risk_score"),
                    record.get("confidence_low_mw"),
                    record.get("confidence_high_mw"),
                    diagnostics_payload,
                    model_version,
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.seasonal_demand_forecast
                (forecast_date, region, sector, scenario_id, base_demand_mw,
                 weather_adjustment_mw, economic_adjustment_mw, holiday_adjustment_mw,
                 final_forecast_mw, peak_risk_score, confidence_low_mw,
                 confidence_high_mw, diagnostics, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    def close(self) -> None:
        if self._external_client:
            return
        try:
            self.client.disconnect()
        except Exception:
            logger.debug("ClickHouse client disconnect failed", exc_info=True)


__all__ = ["SupplyChainPersistence"]
