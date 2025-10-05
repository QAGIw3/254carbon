"""
API Gateway - Stateless REST API for 254Carbon platform

Responsibilities:
- Core REST endpoints (instruments, ticks, curves, fundamentals)
- Authentication/authorization
- Rate limiting
- Caching
- Request routing
- Market adapter integration

Port: 8000
"""
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import microservice clients
from clients.auth_client import verify_token
from clients.entitlements_client import check_entitlement
from clients.metrics_client import track_request

# Import local utilities
from db import get_clickhouse_client, get_postgres_pool
from cache import (
    CacheStrategy,
    cache_response,
    start_cache_warmers_background,
    get_cache_manager,
)
from rate_limiter import add_rate_limiting, limiter, get_rate_limit

# Market adapters are not imported in API Gateway. External data ops are handled by Airflow.

# Feature flags
ENABLE_GRAPHQL = os.getenv("ENABLE_GRAPHQL", "false").lower() == "true"
ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "false").lower() == "true"
ENABLE_RESEARCH = os.getenv("ENABLE_RESEARCH", "false").lower() == "true"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application."""
    logger.info("Starting API Gateway...")

    # Initialize database connections with retry/backoff
    max_attempts = 6
    delay_seconds = 2
    for attempt in range(1, max_attempts + 1):
        try:
            await get_postgres_pool()
            break
        except Exception as e:
            logger.warning(f"Postgres init attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise
            await asyncio.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 30)

    # Start cache warming asynchronously
    start_cache_warmers_background(asyncio.get_running_loop())

    logger.info("API Gateway started successfully")
    yield

    logger.info("Shutting down API Gateway...")


# Create FastAPI application
app = FastAPI(
    title="254Carbon Market Intelligence API",
    description="Stateless REST API for real-time energy and commodity market data",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiting
add_rate_limiting(app)

# No market adapter routers included here

# Include other routers
from gateway.commodity_endpoints import commodity_router
from gateway.commodities_proxy import router as commodities_proxy_router
from gateway.export_endpoints import router as export_router
from gateway.alert_service import alerts_router
from gateway.report_service import create_report_router

# Include commodity endpoints
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))
    app.include_router(commodity_router)
    app.include_router(commodities_proxy_router)
    app.include_router(export_router)
    app.include_router(alerts_router)
    app.include_router(create_report_router())
except Exception as e:
    logger.warning(f"Could not load all gateway routers: {e}")

# Optionally include analytics
if ENABLE_ANALYTICS:
    try:
        from gateway.analytics_endpoints import analytics_router
        app.include_router(analytics_router)
    except Exception as e:
        logger.warning(f"Could not load analytics router: {e}")

# Optionally include research
if ENABLE_RESEARCH:
    try:
        from gateway.research_endpoints import research_router
        app.include_router(research_router)
    except Exception as e:
        logger.warning(f"Could not load research router: {e}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: dict


class InstrumentResponse(BaseModel):
    instrument_id: str
    market: str
    product: str
    location_code: str
    timezone: str
    unit: str
    currency: str


class TickResponse(BaseModel):
    event_time: datetime
    instrument_id: str
    location_code: str
    price_type: str
    value: float
    volume: Optional[float]
    currency: str
    unit: str
    source: str


class CurvePoint(BaseModel):
    delivery_start: date
    delivery_end: date
    tenor_type: str
    price: float
    currency: str
    unit: str


# Health check
@app.get("/health", response_model=HealthResponse)
@limiter.limit(get_rate_limit("public"))
async def health_check(request: Request):
    """Health check endpoint."""
    services = {
        "clickhouse": "healthy",
        "postgres": "healthy",
        "cache": "healthy",
    }
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        services=services,
    )


# Cache management endpoints
@app.get("/api/v1/cache/stats")
@limiter.limit(get_rate_limit("authenticated"))
async def get_cache_stats(request: Request, user=Depends(verify_token)):
    """Get Redis cache statistics."""
    track_request("get_cache_stats")

    try:
        manager = get_cache_manager()
        stats = await manager.get_stats()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error fetching cache stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/cache/warm")
@limiter.limit(get_rate_limit("cache_write"))
async def warm_cache(
    request: Request,
    pattern: Optional[str] = None,
    user=Depends(verify_token),
):
    """Warm cache for specified patterns."""
    track_request("warm_cache")

    try:
        manager = get_cache_manager()

        if pattern:
            success = await manager.warm_cache(pattern)
            return {
                "status": "success" if success else "failed",
                "pattern": pattern,
                "timestamp": datetime.utcnow()
            }
        else:
            results = await manager.warm_all_cache()
            return {
                "status": "completed",
                "results": results,
                "timestamp": datetime.utcnow()
            }
    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Instruments endpoint
@app.get("/api/v1/instruments", response_model=list[InstrumentResponse])
@limiter.limit(get_rate_limit("authenticated"))
@cache_response("instruments", CacheStrategy.SEMI_STATIC)
async def get_instruments(
    request: Request,
    market: Optional[str] = None,
    product: Optional[str] = None,
    user=Depends(verify_token),
):
    """Get available instruments."""
    track_request("get_instruments")
    
    try:
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
            
            instruments = [
                InstrumentResponse(
                    instrument_id=row["instrument_id"],
                    market=row["market"],
                    product=row["product"],
                    location_code=row["location_code"],
                    timezone=row["timezone"],
                    unit=row["unit"],
                    currency=row["currency"],
                )
                for row in rows
            ]
            
            return instruments
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Price ticks endpoint
@app.get("/api/v1/prices/ticks", response_model=list[TickResponse])
@limiter.limit(get_rate_limit("authenticated"))
@cache_response("price_ticks", CacheStrategy.DYNAMIC)
async def get_price_ticks(
    request: Request,
    instrument_id: list[str] = Query(...),
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
    price_type: str = Query("mid"),
    user=Depends(verify_token),
):
    """Get historical price ticks."""
    track_request("get_price_ticks")
    
    # Check entitlements
    for inst_id in instrument_id:
        if not await check_entitlement(user, inst_id, "api"):
            raise HTTPException(
                status_code=403,
                detail=f"Not entitled to API access for {inst_id}",
            )
    
    try:
        ch_client = get_clickhouse_client()
        
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
        
        result = ch_client.execute(
            query,
            {
                "ids": tuple(instrument_id),
                "start": start_time,
                "end": end_time,
                "price_type": price_type,
            },
        )
        
        ticks = [
            TickResponse(
                event_time=row[0],
                instrument_id=row[1],
                location_code=row[2],
                price_type=row[3],
                value=row[4],
                volume=row[5],
                currency=row[6],
                unit=row[7],
                source=row[8],
            )
            for row in result
        ]
        
        return ticks
    except Exception as e:
        logger.error(f"Error fetching ticks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Forward curves endpoint
@app.get("/api/v1/curves/forward", response_model=list[CurvePoint])
@limiter.limit(get_rate_limit("authenticated"))
@cache_response("forward_curves", CacheStrategy.SEMI_STATIC)
async def get_forward_curves(
    request: Request,
    instrument_id: list[str] = Query(...),
    as_of_date: date = Query(...),
    scenario_id: str = Query("BASE"),
    user=Depends(verify_token),
):
    """Get forward curve points."""
    track_request("get_forward_curves")
    
    # Check entitlements
    for inst_id in instrument_id:
        if not await check_entitlement(user, inst_id, "api"):
            raise HTTPException(
                status_code=403,
                detail=f"Not entitled to API access for {inst_id}",
            )
    
    try:
        ch_client = get_clickhouse_client()
        
        query = """
        SELECT 
            delivery_start,
            delivery_end,
            tenor_type,
            price,
            currency,
            unit
        FROM market_intelligence.forward_curve_points
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
        
        points = [
            CurvePoint(
                delivery_start=row[0],
                delivery_end=row[1],
                tenor_type=row[2],
                price=row[3],
                currency=row[4],
                unit=row[5],
            )
            for row in result
        ]
        
        return points
    except Exception as e:
        logger.error(f"Error fetching curves: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Fundamentals endpoint
@app.get("/api/v1/fundamentals")
@limiter.limit(get_rate_limit("authenticated"))
async def get_fundamentals(
    request: Request,
    market: str = Query(...),
    entity_id: str = Query(...),
    variable: str = Query(...),
    start_ts: datetime = Query(...),
    end_ts: datetime = Query(...),
    scenario_id: str = Query("BASE"),
    user=Depends(verify_token),
):
    """Get fundamentals time series."""
    track_request("get_fundamentals")
    
    try:
        ch_client = get_clickhouse_client()
        
        query = """
        SELECT 
            ts,
            entity_id,
            variable,
            value,
            unit,
            scenario_id,
            source
        FROM market_intelligence.fundamentals_series
        WHERE market = %(market)s
          AND entity_id = %(entity_id)s
          AND variable = %(variable)s
          AND ts BETWEEN %(start)s AND %(end)s
          AND scenario_id = %(scenario)s
        ORDER BY ts
        """
        
        result = ch_client.execute(
            query,
            {
                "market": market,
                "entity_id": entity_id,
                "variable": variable,
                "start": start_ts,
                "end": end_ts,
                "scenario": scenario_id,
            },
        )
        
        return [
            {
                "ts": row[0],
                "entity_id": row[1],
                "variable": row[2],
                "value": row[3],
                "unit": row[4],
                "scenario_id": row[5],
                "source": row[6],
            }
            for row in result
        ]
    except Exception as e:
        logger.error(f"Error fetching fundamentals: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Standard error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "about:blank",
            "title": exc.detail,
            "status": exc.status_code,
            "detail": exc.detail,
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

