"""Centralized data access helpers for ML service modules."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
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
