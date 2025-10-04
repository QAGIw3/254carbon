"""Persistence helpers for refining analytics outputs."""

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


class RefiningPersistence:
    """Handles inserts of refining analytics artefacts into ClickHouse."""

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
    # Crack spread optimization
    # ------------------------------------------------------------------
    def persist_crack_optimization(
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
                    record.get("refinery_id"),
                    record.get("crack_type"),
                    record.get("crude_code"),
                    to_float(record.get("gasoline_price")),
                    to_float(record.get("diesel_price")),
                    optional_float(record.get("jet_price")),
                    to_float(record.get("crack_spread")),
                    to_float(record.get("margin_per_bbl")),
                    json_text(record.get("optimal_yields")),
                    json_text(record.get("constraints")),
                    json_dump(record.get("diagnostics")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.refining_crack_optimization
                (as_of_date, region, refinery_id, crack_type, crude_code,
                 gasoline_price, diesel_price, jet_price, crack_spread,
                 margin_per_bbl, optimal_yields, constraints, diagnostics, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Refinery yield modelling
    # ------------------------------------------------------------------
    def persist_refinery_yields(
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
                    record.get("crude_type"),
                    json_text(record.get("process_config")),
                    json_text(record.get("yields")),
                    to_float(record.get("value_per_bbl")),
                    to_float(record.get("operating_cost")),
                    to_float(record.get("net_value")),
                    json_dump(record.get("diagnostics")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.refinery_yield_model
                (as_of_date, crude_type, process_config, yields, value_per_bbl,
                 operating_cost, net_value, diagnostics, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Product demand elasticity
    # ------------------------------------------------------------------
    def persist_product_elasticity(
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
                    record.get("product"),
                    record.get("region"),
                    record.get("method"),
                    to_float(record.get("elasticity")),
                    optional_float(record.get("r_squared")),
                    record.get("own_or_cross"),
                    record.get("product_pair"),
                    to_int(record.get("data_points")),
                    json_dump(record.get("diagnostics")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.product_demand_elasticity
                (as_of_date, product, region, method, elasticity, r_squared,
                 own_or_cross, product_pair, data_points, diagnostics, model_version)
            VALUES
            """,
            rows,
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Transportation fuel substitution metrics
    # ------------------------------------------------------------------
    def persist_transport_substitution(
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
                    record.get("metric"),
                    to_float(record.get("value")),
                    json_text(record.get("details")),
                    record.get("model_version", default_model_version),
                )
            )

        self.client.execute(
            """
            INSERT INTO ch.transport_fuel_substitution
                (as_of_date, region, metric, value, details, model_version)
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


__all__ = ["RefiningPersistence"]
