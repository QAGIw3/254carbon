"""
Curve schema shared across commodity routers.

Represents standardized futures/forward curve points. Commodity routers
populate these using ClickHouse results or synthetic data for dev parity.
"""
from datetime import date
from typing import Optional

from pydantic import BaseModel


class CurvePoint(BaseModel):
    commodity_code: str
    as_of_date: date
    contract_month: str
    settlement_price: float
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    exchange: Optional[str] = None


__all__ = ["CurvePoint"]
