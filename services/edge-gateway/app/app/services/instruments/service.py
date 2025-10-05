"""Instrument service functions for edge gateway."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import asyncpg


async def fetch_instruments(
    pool: asyncpg.Pool,
    *,
    market: Optional[str] = None,
    product: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM pg.instrument WHERE 1=1"
    params: List[Any] = []

    if market:
        params.append(market)
        query += f" AND market = ${len(params)}"

    if product:
        params.append(product)
        query += f" AND product = ${len(params)}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [dict(row) for row in rows]

