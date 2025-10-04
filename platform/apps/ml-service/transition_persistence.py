"""Persistence helpers for transition and carbon analytics outputs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clickhouse_driver import Client

from persistence_utils import (
    as_date,
    json_dump,
    optional_float,
    to_float,
    to_int,
)


logger = logging.getLogger(__name__)


class TransitionPersistence:
    """Handles inserts of transition and carbon analytics artefacts into ClickHouse."""

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
    # Carbon pricing artefacts
    # ------------------------------------------------------------------
    def persist_carbon_price_forecasts(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        default_model_version: str = "v1",
    ) -> int:
        if not records:
            return 0

        rows: List[Tuple[Any, ...]] = []
        for record in records:
            rows.append(
                (
                    as_date(record.get("as_of_date")),
                    record.get("market"),
                    to_int(record.get("horizon_days")),
                    as_date(record.get("forecast_date"), field="forecast_date"),
                    to_float(record.get("forecast_price")),
                    optional_float(record.get("std")),
                    json_dump(record.get("drivers")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.carbon_price_forecast
                (as_of_date, market, horizon_days, forecast_date,
                 forecast_price, std, drivers, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    def persist_compliance_costs(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        default_model_version: str = "v1",
    ) -> int:
        if not records:
            return 0

        rows: List[Tuple[Any, ...]] = []
        for record in records:
            rows.append(
                (
                    as_date(record.get("as_of_date")),
                    record.get("market"),
                    record.get("sector"),
                    to_float(record.get("total_emissions")),
                    to_float(record.get("average_price")),
                    to_float(record.get("cost_per_tonne")),
                    to_float(record.get("total_compliance_cost")),
                    json_dump(record.get("details")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.carbon_compliance_costs
                (as_of_date, market, sector, total_emissions, average_price,
                 cost_per_tonne, total_compliance_cost, details, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    def persist_carbon_leakage_risk(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        default_model_version: str = "v1",
    ) -> int:
        if not records:
            return 0

        rows: List[Tuple[Any, ...]] = []
        for record in records:
            rows.append(
                (
                    as_date(record.get("as_of_date")),
                    record.get("sector"),
                    to_float(record.get("domestic_price")),
                    to_float(record.get("international_price")),
                    to_float(record.get("price_differential")),
                    to_float(record.get("trade_exposure")),
                    to_float(record.get("emissions_intensity")),
                    to_float(record.get("leakage_risk_score")),
                    record.get("risk_level"),
                    json_dump(record.get("details")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.carbon_leakage_risk
                (as_of_date, sector, domestic_price, international_price,
                 price_differential, trade_exposure, emissions_intensity,
                 leakage_risk_score, risk_level, details, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Transition artefacts
    # ------------------------------------------------------------------
    def persist_decarbonization_pathways(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        default_model_version: str = "v1",
    ) -> int:
        if not records:
            return 0

        rows: List[Tuple[Any, ...]] = []
        for record in records:
            rows.append(
                (
                    as_date(record.get("as_of_date")),
                    record.get("sector"),
                    record.get("policy_scenario"),
                    to_int(record.get("target_year")),
                    to_float(record.get("annual_reduction_rate")),
                    to_float(record.get("cumulative_emissions")),
                    1 if bool(record.get("target_achieved")) else 0,
                    json_dump(record.get("emissions_trajectory")),
                    json_dump(record.get("technology_analysis")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.decarbonization_pathways
                (as_of_date, sector, policy_scenario, target_year,
                 annual_reduction_rate, cumulative_emissions, target_achieved,
                 emissions_trajectory, technology_analysis, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    def persist_renewable_adoption_forecast(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        default_model_version: str = "v1",
    ) -> int:
        if not records:
            return 0

        rows: List[Tuple[Any, ...]] = []
        for record in records:
            rows.append(
                (
                    as_date(record.get("as_of_date")),
                    record.get("technology"),
                    as_date(record.get("forecast_year"), field="forecast_year"),
                    to_float(record.get("capacity_gw")),
                    to_float(record.get("policy_support")),
                    json_dump(record.get("economic_multipliers")),
                    json_dump(record.get("assumptions")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.renewable_adoption_forecast
                (as_of_date, technology, forecast_year, capacity_gw,
                 policy_support, economic_multipliers, assumptions, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    def persist_stranded_asset_risk(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        default_model_version: str = "v1",
    ) -> int:
        if not records:
            return 0

        rows: List[Tuple[Any, ...]] = []
        for record in records:
            rows.append(
                (
                    as_date(record.get("as_of_date")),
                    record.get("asset_type"),
                    to_float(record.get("asset_value")),
                    to_float(record.get("carbon_cost_pv")),
                    to_float(record.get("stranded_value")),
                    to_float(record.get("stranded_ratio")),
                    to_int(record.get("remaining_lifetime")),
                    record.get("risk_level"),
                    json_dump(record.get("details")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.stranded_asset_risk
                (as_of_date, asset_type, asset_value, carbon_cost_pv,
                 stranded_value, stranded_ratio, remaining_lifetime,
                 risk_level, details, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    def persist_policy_scenario_impacts(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        default_model_version: str = "v1",
    ) -> int:
        if not records:
            return 0

        rows: List[Tuple[Any, ...]] = []
        for record in records:
            rows.append(
                (
                    as_date(record.get("as_of_date")),
                    record.get("scenario"),
                    record.get("entity"),
                    record.get("metric"),
                    to_float(record.get("value")),
                    json_dump(record.get("details")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.policy_scenario_impacts
                (as_of_date, scenario, entity, metric, value, details, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._external_client:
            return
        try:
            self.client.disconnect()
        except Exception:  # pragma: no cover - graceful shutdown only
            logger.debug("ClickHouse client disconnect failed", exc_info=True)


__all__ = ["TransitionPersistence"]

