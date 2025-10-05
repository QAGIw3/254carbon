"""
Coal-specific schemas.

Provides index price points (API2/Newcastle etc.) and stockpile snapshots.
Stockpiles use approximate weekly cadence for dev parity.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from schemas.pricing import PricePoint


class CoalIndexPoint(PricePoint):
    index_name: Optional[str] = None


class CoalStockpileSnapshot(BaseModel):
    location: str
    timestamp: datetime
    inventory_tons: float
    change_tons: Optional[float] = None
    utilization_pct: Optional[float] = None
    source: Optional[str] = None


__all__ = ["CoalIndexPoint", "CoalStockpileSnapshot"]
