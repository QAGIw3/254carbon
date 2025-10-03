"""
API Gateway with Redis Caching

This is an enhanced version of the gateway with caching layer.
Replace the original main.py with this for production deployment.
"""
import logging
import os
import sys
from datetime import date, datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware

# Add current directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Use absolute imports
from auth import verify_token
from db import get_clickhouse_client, get_postgres_pool
from cache import cache_manager, cache_response
from models import InstrumentResponse, TickResponse, CurvePoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="254Carbon Market Intelligence API - Cached")


# Cached endpoints

@app.get("/api/v1/instruments")
@cache_response("instruments", ttl=3600)  # Cache for 1 hour
async def get_instruments_cached(
    market: Optional[str] = None,
    product: Optional[str] = None,
    user=Depends(verify_token),
):
    """Get available instruments (cached)."""
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        query = "SELECT * FROM pg.instrument WHERE 1=1"
        params = []
        
        if market:
            query += " AND market = $1"
            params.append(market)
        if product:
            query += f" AND product = ${len(params) + 1}"
            params.append(product)
        
        rows = await conn.fetch(query, *params)
        
        return [
            {
                "instrument_id": r["instrument_id"],
                "market": r["market"],
                "product": r["product"],
                "location_code": r["location_code"],
                "timezone": r["timezone"],
                "unit": r["unit"],
                "currency": r["currency"],
            }
            for r in rows
        ]


@app.get("/api/v1/curves/forward")
@cache_response("curves", ttl=1800)  # Cache for 30 minutes
async def get_forward_curves_cached(
    instrument_id: List[str] = Query(...),
    as_of_date: date = Query(...),
    scenario_id: str = Query("BASE"),
    user=Depends(verify_token),
):
    """Get forward curve points (cached)."""
    ch_client = get_clickhouse_client()
    
    query = """
    SELECT 
        delivery_start,
        delivery_end,
        tenor_type,
        price,
        currency,
        unit
    FROM ch.forward_curve_points
    WHERE instrument_id IN %(ids)s
      AND as_of_date = %(date)s
      AND scenario_id = %(scenario)s
    ORDER BY delivery_start
    """
    
    result = ch_client.execute(
        query,
        {
            "ids": tuple(instrument_id),
            "date": as_of_date,
            "scenario": scenario_id,
        },
    )
    
    return [
        {
            "delivery_start": row[0],
            "delivery_end": row[1],
            "tenor_type": row[2],
            "price": row[3],
            "currency": row[4],
            "unit": row[5],
        }
        for row in result
    ]


@app.post("/api/v1/cache/invalidate")
async def invalidate_cache(
    pattern: str,
    user=Depends(verify_token),
):
    """
    Invalidate cache entries matching pattern.
    
    Requires admin role.
    """
    from .auth import has_role
    
    if not has_role(user, "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")
    
    deleted = await cache_manager.delete_pattern(f"254c:{pattern}:*")
    
    return {
        "status": "success",
        "pattern": pattern,
        "keys_deleted": deleted,
    }


@app.get("/api/v1/cache/stats")
async def get_cache_stats(user=Depends(verify_token)):
    """Get cache statistics."""
    stats = await cache_manager.get_stats()
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

