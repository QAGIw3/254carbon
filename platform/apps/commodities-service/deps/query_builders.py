"""
Query builder utilities reused from the gateway service.

These helpers centralize small SQL fragments and parameter assembly for
ClickHouse reads. They intentionally avoid ORM layers to keep performance
and transparency high for time-series queries.
"""
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple


def build_price_query(
    instrument_id: str,
    start: datetime,
    end: datetime,
    price_type: Optional[str] = None,
    location: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 1000,
) -> Tuple[str, Dict[str, Any]]:
    """Build ClickHouse SQL for price ticks with optional filters."""

    clauses: List[str] = [
        "instrument_id = %(instrument_id)s",
        "event_time >= %(start)s",
        "event_time <= %(end)s",
    ]
    params: Dict[str, Any] = {
        "instrument_id": instrument_id,
        "start": start,
        "end": end,
        "limit": limit,
    }

    if price_type:
        clauses.append("price_type = %(price_type)s")
        params["price_type"] = price_type
    if location:
        clauses.append("location_code = %(location)s")
        params["location"] = location
    if source:
        clauses.append("source = %(source)s")
        params["source"] = source

    where_sql = " AND ".join(clauses)
    sql = f"""
    SELECT
        event_time,
        instrument_id,
        location_code,
        price_type,
        value,
        volume,
        currency,
        unit,
        source
    FROM market_intelligence.market_price_ticks
    WHERE {where_sql}
    ORDER BY event_time
    LIMIT %(limit)s
    """
    return sql, params


def build_curve_query(
    commodity_code: str,
    as_of: date,
    exchange: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build SQL for futures curve points by as-of date and optional exchange."""

    clauses = [
        "commodity_code = %(commodity_code)s",
        "as_of_date = %(as_of)s",
    ]
    params: Dict[str, Any] = {
        "commodity_code": commodity_code,
        "as_of": as_of,
    }
    if exchange:
        clauses.append("exchange = %(exchange)s")
        params["exchange"] = exchange

    where_sql = " AND ".join(clauses)
    sql = f"""
    SELECT
        commodity_code,
        as_of_date,
        contract_month,
        settlement_price,
        open_interest,
        volume,
        exchange
    FROM market_intelligence.futures_curves
    WHERE {where_sql}
    ORDER BY contract_month
    """
    return sql, params


def build_latest_snapshot_query(instruments: List[str]) -> Tuple[str, Dict[str, Any]]:
    """Build SQL for latest price snapshots across multiple instruments."""

    sql = """
    SELECT instrument_id, anyLast(value) AS latest_price
    FROM market_intelligence.market_price_ticks
    WHERE instrument_id IN %(instrument_ids)s
    GROUP BY instrument_id
    """
    return sql, {"instrument_ids": tuple(instruments)}


__all__ = [
    "build_price_query",
    "build_curve_query",
    "build_latest_snapshot_query",
]
