"""Price tick service for edge gateway."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List

from clickhouse_driver import Client


async def fetch_price_ticks(
    client: Client,
    *,
    instrument_ids: Iterable[str],
    start_time: str,
    end_time: str,
    price_type: str,
    user: Dict[str, Any],
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            event_time,
            instrument_id,
            location_code,
            price_type,
            value,
            volume,
            currency,
            unit,
            source
        FROM market_intelligence.market_price_ticks
        WHERE instrument_id IN %(ids)s
          AND event_time BETWEEN %(start)s AND %(end)s
          AND price_type = %(price_type)s
        ORDER BY event_time DESC
        LIMIT 10000
    """

    result = client.execute(
        query,
        {
            "ids": tuple(instrument_ids),
            "start": datetime.fromisoformat(start_time),
            "end": datetime.fromisoformat(end_time),
            "price_type": price_type,
        },
    )

    return [
        {
            "event_time": row[0].isoformat(),
            "instrument_id": row[1],
            "location_code": row[2],
            "price_type": row[3],
            "value": row[4],
            "volume": row[5],
            "currency": row[6],
            "unit": row[7],
            "source": row[8],
        }
        for row in result
    ]

