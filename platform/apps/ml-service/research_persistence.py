"""Persistence helpers for commodity research analytics."""

from __future__ import annotations

import logging
from typing import Optional

from clickhouse_driver import Client

from commodity_research_framework import (
    DecompositionResult,
    SupplyDemandResult,
    VolatilityRegimeResult,
    WeatherImpactResult,
)


logger = logging.getLogger(__name__)


class ResearchPersistence:
    """Handles inserts of research analytics into ClickHouse."""

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
    # Commodity decomposition
    # ------------------------------------------------------------------
    def persist_decomposition(self, result: DecompositionResult) -> int:
        rows = result.to_persist_rows()
        if not rows:
            return 0

        self.client.execute(
            """
            INSERT INTO ch.commodity_decomposition
                (snapshot_date, instrument_id, method, trend, seasonal, residual, version, created_at)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Volatility regimes
    # ------------------------------------------------------------------
    def persist_volatility_regimes(self, result: VolatilityRegimeResult) -> int:
        rows = result.to_persist_rows()
        if not rows:
            return 0

        self.client.execute(
            """
            INSERT INTO ch.volatility_regimes
                (date, instrument_id, regime_label, regime_features, method, n_regimes, fit_version, created_at)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Supply / demand metrics
    # ------------------------------------------------------------------
    def persist_supply_demand(self, result: SupplyDemandResult) -> int:
        rows = result.to_persist_rows()
        if not rows:
            return 0

        self.client.execute(
            """
            INSERT INTO ch.supply_demand_metrics
                (date, entity_id, instrument_id, metric_name, metric_value, unit, version, created_at)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Weather impact metrics
    # ------------------------------------------------------------------
    def persist_weather_impact(self, result: WeatherImpactResult) -> int:
        rows = result.to_persist_rows()
        if not rows:
            return 0

        self.client.execute(
            """
            INSERT INTO ch.weather_impact
                (date, entity_id, coef_type, coefficient, r2, p_value, window, model_version, extreme_event_count, diagnostics, method, created_at)
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
        except Exception:  # pragma: no cover - No-op on shutdown failures
            logger.debug("ClickHouse client disconnect failed", exc_info=True)


__all__ = ["ResearchPersistence"]

