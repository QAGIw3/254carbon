"""
Oil commodity endpoints.

Endpoint
- GET /api/v1/commodities/oil/curves: Standardized futures curve for WTI/Brent

Notes
- Queries CH futures_curves with optional exchange filter.
- Falls back to a synthetic curve shaped by a small stochastic slope.
- Uses SEMI_STATIC caching since curves change less frequently intraday.
"""
import logging
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from deps.auth import require_roles
from deps.cache import CacheTTL, cache_response
from deps.db import fetch_clickhouse
from deps.query_builders import build_curve_query
from schemas.curves import CurvePoint
from .synthetic import generate_curve

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/commodities/oil", tags=["oil"])


def _curve_cache_key(
    commodity_code: str,
    as_of_date: date,
    exchange: Optional[str] = None,
    **_: object,
) -> str:
    return f"{commodity_code}:{as_of_date.isoformat()}:{exchange or 'all'}"


@router.get("/curves", response_model=List[CurvePoint])
@cache_response("oil.curves", CacheTTL.SEMI_STATIC, key_builder=_curve_cache_key)
async def get_oil_curves(
    commodity_code: str = Query("CL", description="Commodity code e.g. CL for WTI"),
    as_of_date: date = Query(..., description="Curve as-of date"),
    exchange: Optional[str] = Query(None, description="Optional exchange filter"),
    user: dict = Depends(require_roles("oil_data_access")),
) -> List[CurvePoint]:
    """Return futures curve data for oil benchmarks."""

    del user

    try:
        query, params = build_curve_query(
            commodity_code=commodity_code,
            as_of=as_of_date,
            exchange=exchange,
        )
        rows = await fetch_clickhouse(query, params)
    except Exception as exc:  # pragma: no cover - ClickHouse unavailable
        logger.warning("ClickHouse unavailable for oil curves: %s", exc)
        rows = []

    if rows:
        return [
            CurvePoint(
                commodity_code=row["commodity_code"],
                as_of_date=row["as_of_date"],
                contract_month=row["contract_month"],
                settlement_price=float(row["settlement_price"]),
                open_interest=int(row["open_interest"]) if row.get("open_interest") is not None else None,
                volume=int(row["volume"]) if row.get("volume") is not None else None,
                exchange=row.get("exchange"),
            )
            for row in rows
        ]

    return generate_curve(commodity_code=commodity_code, as_of=as_of_date)


__all__ = ["router"]
