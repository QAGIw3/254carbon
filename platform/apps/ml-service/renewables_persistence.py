"""Persistence helpers for renewables analytics outputs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clickhouse_driver import Client

from persistence_utils import (
    as_date,
    json_dump,
    json_text,
    optional_float,
    to_float,
    to_int,
)


logger = logging.getLogger(__name__)


class RenewablesPersistence:
    """Handles inserts of renewables analytics artefacts into ClickHouse."""

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
    # RIN price forecasts
    # ------------------------------------------------------------------
    def persist_rin_forecast(
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
                    record.get("rin_category"),
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
            INSERT INTO ch.rin_price_forecast
                (as_of_date, rin_category, horizon_days, forecast_date,
                 forecast_price, std, drivers, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Biodiesel vs diesel spread analytics
    # ------------------------------------------------------------------
    def persist_biodiesel_spread(
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
                    record.get("region"),
                    to_float(record.get("mean_gross_spread")),
                    to_float(record.get("mean_net_spread")),
                    to_float(record.get("spread_volatility")),
                    to_int(record.get("arbitrage_opportunities")),
                    json_dump(record.get("diagnostics")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.biodiesel_diesel_spread
                (as_of_date, region, mean_gross_spread, mean_net_spread,
                 spread_volatility, arbitrage_opportunities, diagnostics, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Carbon intensity results
    # ------------------------------------------------------------------
    def persist_carbon_intensity(
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
                    record.get("fuel_type"),
                    record.get("pathway"),
                    to_float(record.get("total_ci")),
                    to_float(record.get("base_emissions")),
                    to_float(record.get("transport_emissions")),
                    to_float(record.get("land_use_emissions")),
                    to_float(record.get("ci_per_mj")),
                    json_text(record.get("assumptions")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.carbon_intensity_results
                (as_of_date, fuel_type, pathway, total_ci, base_emissions,
                 transport_emissions, land_use_emissions, ci_per_mj,
                 assumptions, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Policy impact scenarios (RFS / LCFS)
    # ------------------------------------------------------------------
    def persist_policy_impact(
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
                    record.get("policy"),
                    record.get("entity"),
                    record.get("metric"),
                    to_float(record.get("value")),
                    json_text(record.get("details")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.renewables_policy_impact
                (as_of_date, policy, entity, metric, value, details, model_version)
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


__all__ = ["RenewablesPersistence"]
