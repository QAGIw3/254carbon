"""
Synthetic fallbacks for oil endpoints.

Produces a simple contango/backwardated curve for dev parity when ClickHouse
reads are unavailable or empty. Not intended for analysis, only UX parity.
"""
from datetime import date
from typing import List

import numpy as np

from schemas.curves import CurvePoint


def generate_curve(
    commodity_code: str,
    as_of: date,
    tenors: int = 12,
    base_price: float = 75.0,
) -> List[CurvePoint]:
    """Generate a simple backwardated or contango curve."""

    points: List[CurvePoint] = []
    slope = np.random.normal(-0.5, 0.3)

    for i in range(tenors):
        contract_month = (as_of.replace(day=1) if as_of.day != 1 else as_of)
        month_offset = i
        year = contract_month.year + (contract_month.month - 1 + month_offset) // 12
        month = (contract_month.month - 1 + month_offset) % 12 + 1
        contract_month_str = f"{year:04d}-{month:02d}"
        price = base_price + slope * i + np.random.normal(0, 0.4)
        points.append(
            CurvePoint(
                commodity_code=commodity_code,
                as_of_date=as_of,
                contract_month=contract_month_str,
                settlement_price=round(float(price), 2),
                open_interest=int(np.random.normal(150000, 10000)),
                volume=int(np.random.normal(75000, 8000)),
                exchange="NYMEX",
            )
        )

    return points


__all__ = ["generate_curve"]
