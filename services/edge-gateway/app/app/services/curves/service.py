"""Forward curve service for edge gateway."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, List

from clickhouse_driver import Client


async def fetch_forward_curves(
    client: Client,
    *,
    instrument_ids: Iterable[str],
    as_of_date: str,
    scenario_id: str,
    user: Dict[str, Any],
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            delivery_start,
            delivery_end,
            tenor_type,
            price,
            currency,
            unit
        FROM market_intelligence.forward_curves
        WHERE instrument_id IN %(ids)s
          AND as_of_date = %(as_of)s
          AND scenario_id = %(scenario)s
        ORDER BY delivery_start
        LIMIT 1000
    """

    result = client.execute(
        query,
        {
            "ids": tuple(instrument_ids),
            "as_of": date.fromisoformat(as_of_date),
            "scenario": scenario_id,
        },
    )

    curves: List[Dict[str, Any]] = []
    for row in result:
        curves.append(
            {
                "delivery_start": row[0].isoformat(),
                "delivery_end": row[1].isoformat(),
                "tenor_type": row[2],
                "price": row[3],
                "currency": row[4],
                "unit": row[5],
            }
        )

    return curves

