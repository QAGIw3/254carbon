"""
Battery materials endpoints.

Endpoint
- GET /api/v1/commodities/battery-materials/lithium: Lithium price series and
  a small supply chain snapshot useful for dashboards and demos.

Notes
- Reads from CH when available or synthesizes a stable series.
- Response groups prices and supply chain nodes with `material` echo.
"""
import logging
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from deps.auth import require_roles
from deps.cache import CacheTTL, cache_response
from deps.db import fetch_clickhouse
from deps.query_builders import build_price_query
from .models import BatteryMaterialPrice, Material, SupplyChainNode
from .synthetic import generate_lithium_prices, sample_supply_chain

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/commodities/battery-materials",
    tags=["battery-materials"],
)


class LithiumResponse(BaseModel):
    material: Material
    prices: List[BatteryMaterialPrice]
    supply_chain: List[SupplyChainNode]


def _lithium_cache_key(
    material: Material,
    start_date: date,
    end_date: date,
    exchange: Optional[str] = None,
    **_: object,
) -> str:
    return f"{material.value}:{exchange or 'default'}:{start_date.isoformat()}:{end_date.isoformat()}"


@router.get("/lithium", response_model=LithiumResponse)
@cache_response("battery.lithium", CacheTTL.SEMI_STATIC, key_builder=_lithium_cache_key)
async def get_lithium_market_intel(
    material: Material = Query(Material.LITHIUM_CARBONATE, description="Material to query"),
    start_date: date = Query(...),
    end_date: date = Query(...),
    exchange: Optional[str] = Query(None, description="Exchange or price source"),
    user: dict = Depends(require_roles("battery_materials_access")),
) -> LithiumResponse:
    """Return lithium pricing with a supply chain snapshot."""

    del user

    instrument_id = f"{material.value}_{(exchange or 'spot').upper()}"
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    try:
        query, params = build_price_query(
            instrument_id=instrument_id,
            start=start_dt,
            end=end_dt,
            price_type="spot",
            limit=1000,
        )
        rows = await fetch_clickhouse(query, params)
    except Exception as exc:  # pragma: no cover
        logger.warning("ClickHouse unavailable for lithium prices: %s", exc)
        rows = []

    if rows:
        prices = [
            BatteryMaterialPrice(
                instrument_id=row["instrument_id"],
                material=material,
                timestamp=row["event_time"],
                price=float(row["value"]),
                currency=row.get("currency"),
                unit=row.get("unit"),
                exchange=exchange or row.get("source") or "spot",
                contract_type=row.get("price_type"),
            )
            for row in rows
        ]
    else:
        base_price = 15000 if material == Material.LITHIUM_CARBONATE else 17000
        unit = "USD/tonne"
        prices = generate_lithium_prices(
            material=material,
            start=start_date,
            end=end_date,
            base_price=base_price,
            unit=unit,
            exchange=(exchange or "China_Spot"),
        )

    supply_chain = sample_supply_chain()

    return LithiumResponse(
        material=material,
        prices=prices,
        supply_chain=supply_chain,
    )


__all__ = ["router"]
