"""
Synthetic fallbacks for coal endpoints.

Generates simple daily coal index points and weekly stockpile estimates so UI
and downstream consumers can exercise the endpoint surface offline.
"""
from datetime import date, datetime, timedelta
from typing import List

import numpy as np

from .models import CoalIndexPoint, CoalStockpileSnapshot


def generate_indices(
    index_name: str,
    start: date,
    end: date,
    base_price: float = 150.0,
) -> List[CoalIndexPoint]:
    points: List[CoalIndexPoint] = []
    current = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    while current <= end_dt:
        price = base_price + np.random.normal(0, 1.5)
        points.append(
            CoalIndexPoint(
                instrument_id=index_name,
                index_name=index_name,
                timestamp=current,
                price=round(float(max(price, 50.0)), 2),
                currency="USD",
                unit="$/ton",
                price_type="spot",
            )
        )
        current += timedelta(days=1)
        if len(points) >= 120:
            break

    return points


def generate_stockpiles(
    location: str,
    start: date,
    end: date,
) -> List[CoalStockpileSnapshot]:
    snapshots: List[CoalStockpileSnapshot] = []
    current = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())
    inventory = 5000000.0

    while current <= end_dt:
        change = np.random.normal(-15000, 25000)
        inventory = max(inventory + change, 1000000)
        snapshots.append(
            CoalStockpileSnapshot(
                location=location,
                timestamp=current,
                inventory_tons=round(float(inventory), 0),
                change_tons=round(float(change), 0),
                utilization_pct=round(float(min(max(inventory / 6000000 * 100, 30), 95)), 2),
            )
        )
        current += timedelta(days=7)
        if len(snapshots) >= 52:
            break

    return snapshots


__all__ = ["generate_indices", "generate_stockpiles"]
