"""
Reusable ClickHouse query helpers for commodity endpoints.

Centralizes SQL fragments and validation to avoid duplication across routes.
"""
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple


def build_price_query(
    commodity: str,
    start_date: datetime,
    end_date: datetime,
    price_type: Optional[str] = None,
    location: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 10000,
) -> Tuple[str, Dict[str, Any]]:
    """Build a parameterized ClickHouse query for price time-series by commodity.

    Args:
        commodity: Instrument/commodity identifier
        start_date: Start datetime (inclusive)
        end_date: End datetime (inclusive)
        price_type: Optional price type filter
        location: Optional location code filter
        source: Optional data source filter
        limit: Max rows

    Returns:
        (sql, params) tuple
    """
    clauses: List[str] = [
        "instrument_id = %(commodity)s",
        "event_time >= %(start)s",
        "event_time <= %(end)s",
    ]
    params: Dict[str, Any] = {
        "commodity": commodity,
        "start": start_date,
        "end": end_date,
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
    SELECT event_time, instrument_id, location_code, price_type, value, volume, currency, unit, source
    FROM market_intelligence.market_price_ticks
    WHERE {where_sql}
    ORDER BY event_time
    LIMIT %(limit)s
    """
    params["limit"] = limit
    return sql, params


def build_curve_query(
    commodity: str,
    as_of_date: date,
    exchange: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build ClickHouse query for futures/forward curve by commodity and date."""
    clauses = [
        "commodity_code = %(commodity)s",
        "as_of_date = %(as_of)s",
    ]
    params: Dict[str, Any] = {"commodity": commodity, "as_of": as_of_date}
    if exchange:
        clauses.append("exchange = %(exchange)s")
        params["exchange"] = exchange

    where_sql = " AND ".join(clauses)
    sql = f"""
    SELECT commodity_code, as_of_date, contract_month, settlement_price, open_interest, volume, exchange
    FROM market_intelligence.futures_curves
    WHERE {where_sql}
    ORDER BY contract_month
    """
    return sql, params


def build_latest_snapshot_query(
    instruments: List[str]
) -> Tuple[str, Dict[str, Any]]:
    """Build a query to fetch latest price snapshot for a list of instruments."""
    sql = """
    SELECT instrument_id, anyLast(value) AS latest_price
    FROM market_intelligence.market_price_ticks
    WHERE instrument_id IN %(instruments)s
    GROUP BY instrument_id
    """
    return sql, {"instruments": tuple(instruments)}


