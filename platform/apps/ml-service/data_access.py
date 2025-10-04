"""Centralized data access helpers for ML service modules."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from clickhouse_driver import Client


logger = logging.getLogger(__name__)


def _normalize_timestamp(value: Any) -> datetime:
    """Convert supported timestamp inputs into a timezone-naive UTC datetime."""
    if isinstance(value, datetime):
        return value
    try:
        ts = pd.to_datetime(value)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Unsupported timestamp value: {value}") from exc
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()
    raise ValueError(f"Unsupported timestamp type: {type(ts)!r}")


def _normalize_date(value: Any) -> date:
    """Convert supported inputs into a date object."""
    return _normalize_timestamp(value).date()


def _maybe_load_json(value: Any) -> Any:
    """Best-effort JSON decoding for persistence payloads."""
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class DataAccessLayer:
    """Helper around ClickHouse reads for price, fundamentals, and weather series."""

    def __init__(
        self,
        *,
        ch_client: Optional[Client] = None,
        host: str = "clickhouse",
        port: int = 9000,
        database: str = "ch",
        default_price_type: str = "settle",
    ) -> None:
        self._external_client = ch_client is not None
        self.client = ch_client or Client(host=host, port=port, database=database)
        self.default_price_type = default_price_type
        self._carbon_instrument_map = {
            "eua": "CARBON.EUA",
            "cca": "CARBON.CCA",
            "rggi": "CARBON.RGGI",
            "uk_ets": "CARBON.UKETS",
        }

    # ------------------------------------------------------------------
    # Public price helpers
    # ------------------------------------------------------------------
    def get_price_series(
        self,
        instrument_id: Optional[str],
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        lookback_days: int = 90,
        price_type: Optional[str] = None,
    ) -> pd.Series:
        """Return price series for a single instrument sorted by timestamp."""
        if not instrument_id:
            return pd.Series(dtype=float)

        if start is None and end is None and lookback_days:
            start = datetime.utcnow() - timedelta(days=int(lookback_days))

        filters = {
            "instrument_id": instrument_id,
            "price_type": price_type or self.default_price_type,
        }
        return self._fetch_series(
            table="ch.market_price_ticks",
            time_col="event_time",
            filters=filters,
            start=start,
            end=end,
            order_by="event_time",
            unit_col="unit",
        )

    def get_carbon_price_history(
        self,
        market: str,
        *,
        end: Optional[Any] = None,
        lookback_days: int = 365,
        price_type: Optional[str] = None,
    ) -> pd.Series:
        """Fetch historical carbon prices for a given market."""
        instrument_id = self._resolve_carbon_instrument(market)
        if instrument_id is None:
            logger.warning("Unknown carbon market requested for history: %s", market)
            return pd.Series(dtype=float)

        return self.get_price_series(
            instrument_id,
            end=end,
            lookback_days=lookback_days,
            price_type=price_type or "settle",
        )

    def get_price_dataframe(
        self,
        instrument_ids: Sequence[str],
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        price_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return wide price DataFrame for a set of instruments."""
        instrument_ids = [iid for iid in instrument_ids if iid]
        if not instrument_ids:
            return pd.DataFrame()

        if start is None:
            start = datetime.utcnow() - timedelta(days=365)

        params: Dict[str, Any] = {
            "instrument_ids": tuple(instrument_ids),
            "price_type": price_type or self.default_price_type,
            "start": _normalize_timestamp(start),
        }
        query = [
            "SELECT instrument_id, event_time, value",
            " FROM ch.market_price_ticks",
            " WHERE instrument_id IN %(instrument_ids)s",
            "   AND price_type = %(price_type)s",
            "   AND event_time >= %(start)s",
        ]
        if end is not None:
            params["end"] = _normalize_timestamp(end)
            query.append("   AND event_time <= %(end)s")
        query.append(" ORDER BY event_time, instrument_id")

        rows = self.client.execute("".join(query), params)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["instrument_id", "event_time", "value"])
        df["event_time"] = pd.to_datetime(df["event_time"])
        pivot = df.pivot_table(
            index="event_time",
            columns="instrument_id",
            values="value",
        ).sort_index()
        return pivot

    def get_return_series(
        self,
        instrument_id: Optional[str],
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        lookback_days: int = 365,
        price_type: Optional[str] = None,
    ) -> pd.Series:
        """Fetch log returns for a given instrument."""
        prices = self.get_price_series(
            instrument_id,
            start=start,
            end=end,
            lookback_days=lookback_days,
            price_type=price_type,
        )
        if prices.empty:
            return prices
        return prices.sort_index().pct_change().dropna()

    def get_infrastructure_assets(
        self,
        asset_types: Optional[Sequence[str]] = None,
        *,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch infrastructure asset inventory for stranded asset analytics."""

        columns = [
            "asset_type",
            "region",
            "asset_value",
            "emissions_intensity",
            "remaining_lifetime",
            "metadata",
        ]

        query = [
            "SELECT asset_type, region, asset_value, emissions_intensity, remaining_lifetime, metadata",
            " FROM ch.infrastructure_assets",
        ]
        params: Dict[str, Any] = {}
        if asset_types:
            params["asset_types"] = tuple({atype.lower() for atype in asset_types})
            query.append(" WHERE lower(asset_type) IN %(asset_types)s")
        if limit is not None:
            params["limit"] = int(limit)
            query.append(" LIMIT %(limit)s")

        try:
            rows = self.client.execute("".join(query), params if params else None)
        except Exception:
            logger.debug("Infrastructure assets lookup failed", exc_info=True)
            return pd.DataFrame(columns=columns)

        if not rows:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(rows, columns=columns)
        df["asset_type"] = df["asset_type"].str.lower()
        return df

    # ------------------------------------------------------------------
    # Fundamentals and weather helpers
    # ------------------------------------------------------------------
    def get_fundamental_series(
        self,
        entity_id: Optional[str],
        variable: Optional[str],
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        lookback_days: int = 180,
        scenario_id: str = "BASE",
    ) -> pd.Series:
        if not entity_id or not variable:
            return pd.Series(dtype=float)

        if start is None and end is None:
            start = datetime.utcnow() - timedelta(days=int(lookback_days))

        filters = {
            "entity_id": entity_id,
            "variable": variable,
            "scenario_id": scenario_id,
        }
        return self._fetch_series(
            table="ch.fundamentals_series",
            time_col="ts",
            filters=filters,
            start=start,
            end=end,
            order_by="ts",
            unit_col="unit",
        )

    def get_weather_series(
        self,
        entity_id: Optional[str],
        variable: Optional[str],
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        lookback_days: int = 365,
        scenario_id: str = "BASE",
        table: str = "ch.weather_observations",
    ) -> pd.Series:
        """Fetch weather series with fallback to fundamentals table if needed."""
        if not entity_id or not variable:
            return pd.Series(dtype=float)

        if start is None and end is None:
            start = datetime.utcnow() - timedelta(days=int(lookback_days))

        # Primary weather observations table
        filters = {
            "entity_id": entity_id,
            "variable": variable,
        }
        series = self._fetch_series(
            table=table,
            time_col="ts",
            filters=filters,
            start=start,
            end=end,
            order_by="ts",
            required=False,
            unit_col="unit",
        )

        if not series.empty:
            return series

        # Fallback to fundamentals table tagged as weather data
        fallback_filters = {
            "entity_id": entity_id,
            "variable": variable,
            "scenario_id": scenario_id,
        }
        return self._fetch_series(
            table="ch.fundamentals_series",
            time_col="ts",
            filters=fallback_filters,
            start=start,
            end=end,
            order_by="ts",
            unit_col="unit",
        )

    # ------------------------------------------------------------------
    # Refining and renewables analytics fetchers
    # ------------------------------------------------------------------
    def fetch_refining_crack_optimization(
        self,
        *,
        region: Optional[str] = None,
        crack_type: Optional[str] = None,
        crude_code: Optional[str] = None,
        refinery_id: Optional[str] = None,
        as_of_date: Optional[Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        columns = [
            "as_of_date",
            "region",
            "refinery_id",
            "crack_type",
            "crude_code",
            "gasoline_price",
            "diesel_price",
            "jet_price",
            "crack_spread",
            "margin_per_bbl",
            "optimal_yields",
            "constraints",
            "diagnostics",
            "model_version",
            "created_at",
        ]
        query_parts = [
            "SELECT as_of_date, region, refinery_id, crack_type, crude_code,",
            "       gasoline_price, diesel_price, jet_price, crack_spread,",
            "       margin_per_bbl, optimal_yields, constraints, diagnostics,",
            "       model_version, created_at",
            " FROM ch.refining_crack_optimization",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        if as_of_date is not None:
            params["as_of_date"] = _normalize_date(as_of_date)
            query_parts.append(" AND as_of_date = %(as_of_date)s")
        if start_date is not None:
            params["start_date"] = _normalize_date(start_date)
            query_parts.append(" AND as_of_date >= %(start_date)s")
        if end_date is not None:
            params["end_date"] = _normalize_date(end_date)
            query_parts.append(" AND as_of_date <= %(end_date)s")
        if region:
            params["region"] = region
            query_parts.append(" AND region = %(region)s")
        if crack_type:
            params["crack_type"] = crack_type
            query_parts.append(" AND crack_type = %(crack_type)s")
        if crude_code:
            params["crude_code"] = crude_code
            query_parts.append(" AND crude_code = %(crude_code)s")
        if refinery_id:
            params["refinery_id"] = refinery_id
            query_parts.append(" AND refinery_id = %(refinery_id)s")

        query_parts.append(" ORDER BY as_of_date DESC, region, crack_type")
        if limit:
            query_parts.append(f" LIMIT {int(limit)}")

        rows = self.client.execute("".join(query_parts), params)
        return self._rows_to_dicts(
            rows,
            columns,
            json_fields=("optimal_yields", "constraints", "diagnostics"),
        )

    def fetch_refinery_yield_results(
        self,
        *,
        crude_type: Optional[str] = None,
        as_of_date: Optional[Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        columns = [
            "as_of_date",
            "crude_type",
            "process_config",
            "yields",
            "value_per_bbl",
            "operating_cost",
            "net_value",
            "diagnostics",
            "model_version",
            "created_at",
        ]
        query_parts = [
            "SELECT as_of_date, crude_type, process_config, yields,",
            "       value_per_bbl, operating_cost, net_value, diagnostics,",
            "       model_version, created_at",
            " FROM ch.refinery_yield_model",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        if crude_type:
            params["crude_type"] = crude_type
            query_parts.append(" AND crude_type = %(crude_type)s")
        if as_of_date is not None:
            params["as_of_date"] = _normalize_date(as_of_date)
            query_parts.append(" AND as_of_date = %(as_of_date)s")
        if start_date is not None:
            params["start_date"] = _normalize_date(start_date)
            query_parts.append(" AND as_of_date >= %(start_date)s")
        if end_date is not None:
            params["end_date"] = _normalize_date(end_date)
            query_parts.append(" AND as_of_date <= %(end_date)s")

        query_parts.append(" ORDER BY as_of_date DESC, crude_type")
        if limit:
            query_parts.append(f" LIMIT {int(limit)}")

        rows = self.client.execute("".join(query_parts), params)
        return self._rows_to_dicts(
            rows,
            columns,
            json_fields=("process_config", "yields", "diagnostics"),
        )

    def fetch_product_elasticities(
        self,
        *,
        product: Optional[str] = None,
        region: Optional[str] = None,
        own_or_cross: Optional[str] = None,
        method: Optional[str] = None,
        as_of_date: Optional[Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        columns = [
            "as_of_date",
            "product",
            "region",
            "method",
            "elasticity",
            "r_squared",
            "own_or_cross",
            "product_pair",
            "data_points",
            "diagnostics",
            "model_version",
            "created_at",
        ]
        query_parts = [
            "SELECT as_of_date, product, region, method, elasticity, r_squared,",
            "       own_or_cross, product_pair, data_points, diagnostics,",
            "       model_version, created_at",
            " FROM ch.product_demand_elasticity",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        if product:
            params["product"] = product
            query_parts.append(" AND product = %(product)s")
        if region:
            params["region"] = region
            query_parts.append(" AND region = %(region)s")
        if own_or_cross:
            params["own_or_cross"] = own_or_cross
            query_parts.append(" AND own_or_cross = %(own_or_cross)s")
        if method:
            params["method"] = method
            query_parts.append(" AND method = %(method)s")
        if as_of_date is not None:
            params["as_of_date"] = _normalize_date(as_of_date)
            query_parts.append(" AND as_of_date = %(as_of_date)s")
        if start_date is not None:
            params["start_date"] = _normalize_date(start_date)
            query_parts.append(" AND as_of_date >= %(start_date)s")
        if end_date is not None:
            params["end_date"] = _normalize_date(end_date)
            query_parts.append(" AND as_of_date <= %(end_date)s")

        query_parts.append(" ORDER BY as_of_date DESC, product, region")
        if limit:
            query_parts.append(f" LIMIT {int(limit)}")

        rows = self.client.execute("".join(query_parts), params)
        return self._rows_to_dicts(
            rows,
            columns,
            json_fields=("diagnostics",),
        )

    def fetch_transport_substitution_metrics(
        self,
        *,
        region: Optional[str] = None,
        metric: Optional[str] = None,
        as_of_date: Optional[Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        columns = [
            "as_of_date",
            "region",
            "metric",
            "value",
            "details",
            "model_version",
            "created_at",
        ]
        query_parts = [
            "SELECT as_of_date, region, metric, value, details, model_version, created_at",
            " FROM ch.transport_fuel_substitution",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        if region:
            params["region"] = region
            query_parts.append(" AND region = %(region)s")
        if metric:
            params["metric"] = metric
            query_parts.append(" AND metric = %(metric)s")
        if as_of_date is not None:
            params["as_of_date"] = _normalize_date(as_of_date)
            query_parts.append(" AND as_of_date = %(as_of_date)s")
        if start_date is not None:
            params["start_date"] = _normalize_date(start_date)
            query_parts.append(" AND as_of_date >= %(start_date)s")
        if end_date is not None:
            params["end_date"] = _normalize_date(end_date)
            query_parts.append(" AND as_of_date <= %(end_date)s")

        query_parts.append(" ORDER BY as_of_date DESC, region, metric")
        if limit:
            query_parts.append(f" LIMIT {int(limit)}")

        rows = self.client.execute("".join(query_parts), params)
        return self._rows_to_dicts(
            rows,
            columns,
            json_fields=("details",),
        )

    def fetch_rin_price_forecasts(
        self,
        *,
        rin_category: Optional[str] = None,
        as_of_date: Optional[Any] = None,
        forecast_date: Optional[Any] = None,
        horizon_days: Optional[int] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        columns = [
            "as_of_date",
            "rin_category",
            "horizon_days",
            "forecast_date",
            "forecast_price",
            "std",
            "drivers",
            "model_version",
            "created_at",
        ]
        query_parts = [
            "SELECT as_of_date, rin_category, horizon_days, forecast_date,",
            "       forecast_price, std, drivers, model_version, created_at",
            " FROM ch.rin_price_forecast",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        if rin_category:
            params["rin_category"] = rin_category
            query_parts.append(" AND rin_category = %(rin_category)s")
        if as_of_date is not None:
            params["as_of_date"] = _normalize_date(as_of_date)
            query_parts.append(" AND as_of_date = %(as_of_date)s")
        if forecast_date is not None:
            params["forecast_date"] = _normalize_date(forecast_date)
            query_parts.append(" AND forecast_date = %(forecast_date)s")
        if horizon_days is not None:
            params["horizon_days"] = int(horizon_days)
            query_parts.append(" AND horizon_days = %(horizon_days)s")

        query_parts.append(" ORDER BY forecast_date DESC, rin_category, horizon_days")
        if limit:
            query_parts.append(f" LIMIT {int(limit)}")

        rows = self.client.execute("".join(query_parts), params)
        return self._rows_to_dicts(
            rows,
            columns,
            json_fields=("drivers",),
        )

    def fetch_biodiesel_spreads(
        self,
        *,
        region: Optional[str] = None,
        as_of_date: Optional[Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        columns = [
            "as_of_date",
            "region",
            "mean_gross_spread",
            "mean_net_spread",
            "spread_volatility",
            "arbitrage_opportunities",
            "diagnostics",
            "model_version",
            "created_at",
        ]
        query_parts = [
            "SELECT as_of_date, region, mean_gross_spread, mean_net_spread,",
            "       spread_volatility, arbitrage_opportunities, diagnostics,",
            "       model_version, created_at",
            " FROM ch.biodiesel_diesel_spread",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        if region:
            params["region"] = region
            query_parts.append(" AND region = %(region)s")
        if as_of_date is not None:
            params["as_of_date"] = _normalize_date(as_of_date)
            query_parts.append(" AND as_of_date = %(as_of_date)s")
        if start_date is not None:
            params["start_date"] = _normalize_date(start_date)
            query_parts.append(" AND as_of_date >= %(start_date)s")
        if end_date is not None:
            params["end_date"] = _normalize_date(end_date)
            query_parts.append(" AND as_of_date <= %(end_date)s")

        query_parts.append(" ORDER BY as_of_date DESC, region")
        if limit:
            query_parts.append(f" LIMIT {int(limit)}")

        rows = self.client.execute("".join(query_parts), params)
        return self._rows_to_dicts(
            rows,
            columns,
            json_fields=("diagnostics",),
        )

    def fetch_carbon_intensity_results(
        self,
        *,
        fuel_type: Optional[str] = None,
        pathway: Optional[str] = None,
        as_of_date: Optional[Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        columns = [
            "as_of_date",
            "fuel_type",
            "pathway",
            "total_ci",
            "base_emissions",
            "transport_emissions",
            "land_use_emissions",
            "ci_per_mj",
            "assumptions",
            "model_version",
            "created_at",
        ]
        query_parts = [
            "SELECT as_of_date, fuel_type, pathway, total_ci, base_emissions,",
            "       transport_emissions, land_use_emissions, ci_per_mj,",
            "       assumptions, model_version, created_at",
            " FROM ch.carbon_intensity_results",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        if fuel_type:
            params["fuel_type"] = fuel_type
            query_parts.append(" AND fuel_type = %(fuel_type)s")
        if pathway:
            params["pathway"] = pathway
            query_parts.append(" AND pathway = %(pathway)s")
        if as_of_date is not None:
            params["as_of_date"] = _normalize_date(as_of_date)
            query_parts.append(" AND as_of_date = %(as_of_date)s")
        if start_date is not None:
            params["start_date"] = _normalize_date(start_date)
            query_parts.append(" AND as_of_date >= %(start_date)s")
        if end_date is not None:
            params["end_date"] = _normalize_date(end_date)
            query_parts.append(" AND as_of_date <= %(end_date)s")

        query_parts.append(" ORDER BY as_of_date DESC, fuel_type, pathway")
        if limit:
            query_parts.append(f" LIMIT {int(limit)}")

        rows = self.client.execute("".join(query_parts), params)
        return self._rows_to_dicts(
            rows,
            columns,
            json_fields=("assumptions",),
        )

    def fetch_policy_impact_metrics(
        self,
        *,
        policy: Optional[str] = None,
        entity: Optional[str] = None,
        metric: Optional[str] = None,
        as_of_date: Optional[Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        columns = [
            "as_of_date",
            "policy",
            "entity",
            "metric",
            "value",
            "details",
            "model_version",
            "created_at",
        ]
        query_parts = [
            "SELECT as_of_date, policy, entity, metric, value, details,",
            "       model_version, created_at",
            " FROM ch.renewables_policy_impact",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        if policy:
            params["policy"] = policy
            query_parts.append(" AND policy = %(policy)s")
        if entity:
            params["entity"] = entity
            query_parts.append(" AND entity = %(entity)s")
        if metric:
            params["metric"] = metric
            query_parts.append(" AND metric = %(metric)s")
        if as_of_date is not None:
            params["as_of_date"] = _normalize_date(as_of_date)
            query_parts.append(" AND as_of_date = %(as_of_date)s")
        if start_date is not None:
            params["start_date"] = _normalize_date(start_date)
            query_parts.append(" AND as_of_date >= %(start_date)s")
        if end_date is not None:
            params["end_date"] = _normalize_date(end_date)
            query_parts.append(" AND as_of_date <= %(end_date)s")

        query_parts.append(" ORDER BY as_of_date DESC, policy, entity, metric")
        if limit:
            query_parts.append(f" LIMIT {int(limit)}")

        rows = self.client.execute("".join(query_parts), params)
        return self._rows_to_dicts(
            rows,
            columns,
            json_fields=("details",),
        )

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._external_client:
            return
        try:
            self.client.disconnect()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rows_to_dicts(
        self,
        rows: Sequence[Sequence[Any]],
        columns: Sequence[str],
        *,
        json_fields: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        json_cols = set(json_fields)
        records: List[Dict[str, Any]] = []
        for row in rows:
            record: Dict[str, Any] = {}
            for idx, column in enumerate(columns):
                value = row[idx]
                if column in json_cols:
                    record[column] = _maybe_load_json(value)
                else:
                    record[column] = value
            records.append(record)
        return records

    def _resolve_carbon_instrument(self, market: Optional[str]) -> Optional[str]:
        if market is None:
            return None
        return self._carbon_instrument_map.get(market.lower())

    def _fetch_series(
        self,
        *,
        table: str,
        time_col: str,
        filters: Dict[str, Any],
        start: Optional[Any],
        end: Optional[Any],
        order_by: str,
        required: bool = True,
        value_col: str = "value",
        unit_col: Optional[str] = None,
    ) -> pd.Series:
        select_cols = [time_col, value_col]
        if unit_col:
            select_cols.append(unit_col)

        query_parts = [
            f"SELECT {', '.join(select_cols)}",
            f" FROM {table}",
            " WHERE 1 = 1",
        ]
        params: Dict[str, Any] = {}

        for key, value in filters.items():
            if value is None:
                continue
            param_key = f"param_{key}"
            query_parts.append(f" AND {key} = %({param_key})s")
            params[param_key] = value

        if start is not None:
            params["start"] = _normalize_timestamp(start)
            query_parts.append(f" AND {time_col} >= %(start)s")

        if end is not None:
            params["end"] = _normalize_timestamp(end)
            query_parts.append(f" AND {time_col} <= %(end)s")

        query_parts.append(f" ORDER BY {order_by}")

        rows: List[Tuple[Any, ...]] = self.client.execute("".join(query_parts), params)
        if not rows:
            if required:
                logger.debug(
                    "No rows returned for %s where %s", table, filters
                )
            return pd.Series(dtype=float)

        idx = pd.to_datetime([row[0] for row in rows])
        values = [row[1] for row in rows]
        series = pd.Series(values, index=idx, dtype=float).sort_index()

        if unit_col:
            try:
                unit_index = select_cols.index(unit_col)
            except ValueError:
                unit_index = -1
            if unit_index >= 0:
                units = [row[unit_index] for row in rows if len(row) > unit_index]
                unit_values = [u for u in units if u not in (None, "")]
                if unit_values:
                    try:
                        series.attrs["unit"] = str(unit_values[0])
                    except Exception:  # pragma: no cover - attrs guard
                        pass
        return series


__all__ = ["DataAccessLayer"]
