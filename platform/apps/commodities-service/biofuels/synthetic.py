"""
Synthetic fallbacks for biofuels endpoints.

Provides simple daily RIN price series to keep the API usable offline.
Intended only for development parity and smoke testing.
"""
from datetime import date, datetime, timedelta
from typing import List

import numpy as np

from .models import RINPricePoint


def generate_rin_prices(
    rin_type: str,
    start: date,
    end: date,
    base_price: float = 1.6,
) -> List[RINPricePoint]:
    points: List[RINPricePoint] = []
    current = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    while current <= end_dt:
        price = base_price + np.random.normal(0, 0.05)
        points.append(
            RINPricePoint(
                instrument_id=rin_type,
                rin_type=rin_type,
                timestamp=current,
                price=round(float(max(price, 0.5)), 3),
                currency="USD",
                unit="USD/RIN",
                price_type="spot",
            )
        )
        current += timedelta(days=1)
        if len(points) >= 180:
            break

    return points


__all__ = ["generate_rin_prices"]
