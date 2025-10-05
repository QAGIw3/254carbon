"""
Coal commodity endpoints.

Endpoints
- GET /api/v1/commodities/coal/indices: Coal benchmark indices
- GET /api/v1/commodities/coal/stockpiles: Stockpile estimates for a location

Notes
- Prefers ClickHouse reads with structured metadata columns.
- Synthetic fallbacks ensure development and demos are unblocked.
"""
import logging
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from deps.auth import require_roles
from deps.cache import CacheTTL, cache_response
from deps.db import fetch_clickhouse
from .models import CoalIndexPoint, CoalStockpileSnapshot
from .synthetic import generate_indices, generate_stockpiles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/commodities/coal", tags=["coal"])


def _indices_cache_key(
    index_name: str,
    start_date: date,
    end_date: date,
    **_: object,
) -> str:
    return f"{index_name}:{start_date.isoformat()}:{end_date.isoformat()}"


@router.get("/indices", response_model=List[CoalIndexPoint])
@cache_response("coal.indices", CacheTTL.SEMI_STATIC, key_builder=_indices_cache_key)
async def get_coal_indices(
    index_name: str = Query("API2", description="Coal index name"),
    start_date: date = Query(...),
    end_date: date = Query(...),
    user: dict = Depends(require_roles("coal_data_access")),
) -> List[CoalIndexPoint]:
    """Return coal index price history."""

    del user

    try:
        query = """
        SELECT
            event_time AS timestamp,
            instrument_id,
            value AS price,
            currency,
            unit,
            location_code,
            source
        FROM market_intelligence.market_price_ticks
        WHERE product = 'coal_index'
          AND instrument_id = %(index)s
          AND event_time >= %(start)s
          AND event_time <= %(end)s
        ORDER BY event_time
        """
        params = {
            "index": index_name,
            "start": datetime.combine(start_date, datetime.min.time()),
            "end": datetime.combine(end_date, datetime.max.time()),
        }
        rows = await fetch_clickhouse(query, params)
    except Exception as exc:  # pragma: no cover
        logger.warning("ClickHouse unavailable for coal indices: %s", exc)
        rows = []

    if rows:
        return [
            CoalIndexPoint(
                instrument_id=row["instrument_id"],
                index_name=index_name,
                timestamp=row["timestamp"],
                price=float(row.get("price") or 0.0),
                currency=row.get("currency"),
                unit=row.get("unit"),
                location=row.get("location_code"),
                source=row.get("source"),
                price_type="spot",
            )
            for row in rows
        ]

    return generate_indices(index_name=index_name, start=start_date, end=end_date)


def _stockpile_cache_key(
    location: str,
    start_date: date,
    end_date: date,
    **_: object,
) -> str:
    return f"{location}:{start_date.isoformat()}:{end_date.isoformat()}"


@router.get("/stockpiles", response_model=List[CoalStockpileSnapshot])
@cache_response("coal.stockpiles", CacheTTL.SEMI_STATIC, key_builder=_stockpile_cache_key)
async def get_coal_stockpiles(
    location: str = Query("US_Ports", description="Stockpile location"),
    start_date: date = Query(...),
    end_date: date = Query(...),
    user: dict = Depends(require_roles("coal_data_access")),
) -> List[CoalStockpileSnapshot]:
    """Return coal stockpile estimates."""

    del user

    try:
        query = """
        SELECT
            event_time AS timestamp,
            location_code AS location,
            value AS inventory_tons,
            metadata:change AS change_tons,
            metadata:utilization AS utilization_pct,
            source
        FROM market_intelligence.market_price_ticks
        WHERE product = 'coal_stockpile'
          AND location_code = %(location)s
          AND event_time >= %(start)s
          AND event_time <= %(end)s
        ORDER BY event_time
        """
        params = {
            "location": location,
            "start": datetime.combine(start_date, datetime.min.time()),
            "end": datetime.combine(end_date, datetime.max.time()),
        }
        rows = await fetch_clickhouse(query, params)
    except Exception as exc:  # pragma: no cover
        logger.warning("ClickHouse unavailable for coal stockpiles: %s", exc)
        rows = []

    if rows:
        return [
            CoalStockpileSnapshot(
                location=row.get("location", location),
                timestamp=row["timestamp"],
                inventory_tons=float(row.get("inventory_tons") or 0.0),
                change_tons=float(row.get("change_tons") or 0.0) if row.get("change_tons") is not None else None,
                utilization_pct=float(row.get("utilization_pct") or 0.0) if row.get("utilization_pct") is not None else None,
                source=row.get("source"),
            )
            for row in rows
        ]

    return generate_stockpiles(location=location, start=start_date, end=end_date)


__all__ = ["router"]
