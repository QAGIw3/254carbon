"""
Biofuels endpoints.

Endpoint
- GET /api/v1/commodities/biofuels/rin-prices: RIN price series by category.

Notes
- CH-backed when available; otherwise synthetic series. Cached SEMI_STATIC.
- Role requirement: `biofuels_data_access`.
"""
import logging
from datetime import date, datetime
from typing import List

from fastapi import APIRouter, Depends, Query

from deps.auth import require_roles
from deps.cache import CacheTTL, cache_response
from deps.db import fetch_clickhouse
from .models import RINPricePoint
from .synthetic import generate_rin_prices

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/commodities/biofuels", tags=["biofuels"])


def _rin_cache_key(rin_type: str, start_date: date, end_date: date, **_: object) -> str:
    return f"{rin_type}:{start_date.isoformat()}:{end_date.isoformat()}"


@router.get("/rin-prices", response_model=List[RINPricePoint])
@cache_response("biofuels.rins", CacheTTL.SEMI_STATIC, key_builder=_rin_cache_key)
async def get_rin_prices(
    rin_type: str = Query("D4", description="RIN category e.g. D4, D6"),
    start_date: date = Query(...),
    end_date: date = Query(...),
    user: dict = Depends(require_roles("biofuels_data_access")),
) -> List[RINPricePoint]:
    """Return Renewable Identification Number price time-series."""

    del user

    try:
        query = """
        SELECT
            event_time AS timestamp,
            instrument_id,
            value AS price,
            currency,
            unit,
            source
        FROM market_intelligence.market_price_ticks
        WHERE product = 'rin_price'
          AND instrument_id = %(rin_type)s
          AND event_time >= %(start)s
          AND event_time <= %(end)s
        ORDER BY event_time
        """
        rows = await fetch_clickhouse(
            query,
            {
                "rin_type": rin_type,
                "start": datetime.combine(start_date, datetime.min.time()),
                "end": datetime.combine(end_date, datetime.max.time()),
            },
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("ClickHouse unavailable for RIN prices: %s", exc)
        rows = []

    if rows:
        return [
            RINPricePoint(
                instrument_id=row["instrument_id"],
                rin_type=rin_type,
                timestamp=row["timestamp"],
                price=float(row.get("price") or 0.0),
                currency=row.get("currency"),
                unit=row.get("unit"),
                source=row.get("source"),
                price_type="spot",
            )
            for row in rows
        ]

    return generate_rin_prices(rin_type=rin_type, start=start_date, end=end_date)


__all__ = ["router"]
