"""
API Gateway Service
FastAPI application with OIDC integration, core endpoints, and WebSocket streaming.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .auth import verify_token, has_permission
from .db import get_clickhouse_client, get_postgres_pool
from .entitlements import check_entitlement
from .metrics import track_request, track_latency
from .stream import StreamManager
from .websocket_auth import verify_ws_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stream manager for WebSocket connections
stream_manager = StreamManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application."""
    logger.info("Starting API Gateway...")
    # Initialize database connections
    await get_postgres_pool()
    logger.info("API Gateway started successfully")
    yield
    logger.info("Shutting down API Gateway...")
    # Cleanup connections
    await stream_manager.shutdown()


# Create FastAPI application
app = FastAPI(
    title="254Carbon Market Intelligence API",
    description="Real-time energy and commodity market data, curves, and forecasts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Web Hub
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
async def health_check():
    """Health check endpoint."""
    services = {
        "clickhouse": "healthy",
        "postgres": "healthy",
        "kafka": "healthy",
    }
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        services=services,
    )


# Instruments endpoint
@app.get("/api/v1/instruments", response_model=list[InstrumentResponse])
async def get_instruments(
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
async def get_price_ticks(
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
        FROM ch.market_price_ticks
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
async def get_forward_curves(
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
async def get_fundamentals(
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
        FROM ch.fundamentals_series
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


# WebSocket streaming endpoint
@app.websocket("/api/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time price streaming via WebSocket."""
    await websocket.accept()

    instrument_ids = []

    try:
        # Authenticate
        auth_msg = await websocket.receive_json()

        if auth_msg.get("type") == "subscribe":
            instrument_ids = auth_msg.get("instruments", [])
            api_key = auth_msg.get("api_key")

            # Production: validate JWT; Dev: allow dev-key
            if os.getenv("LOCAL_DEV", "true") != "true":
                try:
                    await verify_ws_token(api_key)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Unauthorized"})
                    await websocket.close()
                    return

            # Optional entitlement checks in production
            if os.getenv("LOCAL_DEV", "true") != "true":
                # verify_ws_token already validated; reuse its claims for entitlements
                try:
                    user_claims = await verify_ws_token(api_key)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Unauthorized"})
                    await websocket.close()
                    return

                for inst_id in instrument_ids:
                    entitled = await check_entitlement(user_claims, inst_id, "stream")
                    if not entitled:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Not entitled to stream {inst_id}"
                        })
                        await websocket.close()
                        return

            # Register connection
            await stream_manager.register(websocket, instrument_ids)

            # Send confirmation
            await websocket.send_json({
                "type": "subscribed",
                "instruments": instrument_ids,
                "message": f"Subscribed to {len(instrument_ids)} instruments"
            })

            # Start streaming mock data for local development
            if os.getenv("LOCAL_DEV", "true") == "true":
                await stream_mock_data(websocket, instrument_ids)
            else:
                # In production, would stream from Kafka
                await stream_kafka_data(websocket, instrument_ids)

        else:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid subscription message"
            })

    except WebSocketDisconnect:
        await stream_manager.unregister(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await stream_manager.unregister(websocket)


async def stream_mock_data(websocket: WebSocket, instrument_ids: list[str]):
    """Stream mock price data for local development."""
    import random
    import json

    while True:
        try:
            # Generate mock price updates
            for instrument_id in instrument_ids:
                # Generate realistic price based on instrument
                if "MISO" in instrument_id:
                    base_price = 35.0
                elif "PJM" in instrument_id:
                    base_price = 40.0
                elif "CAISO" in instrument_id:
                    base_price = 45.0
                else:
                    base_price = 40.0

                # Add some random variation
                price = base_price + random.uniform(-2, 2)
                price = max(0, price)  # Ensure non-negative

                price_update = {
                    "type": "price_update",
                    "data": {
                        "instrument_id": instrument_id,
                        "value": round(price, 2),
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "mock",
                        "market": "power",
                        "product": "lmp"
                    }
                }

                await websocket.send_json(price_update)

            # Send updates every 5 seconds
            await asyncio.sleep(5)

        except WebSocketDisconnect:
            break
        except Exception as e:
            logger.error(f"Error streaming mock data: {e}")
            break


async def stream_kafka_data(websocket: WebSocket, instrument_ids: list[str]):
    """Stream real price data from Kafka (placeholder)."""
    # In production, this would consume from Kafka and filter for subscribed instruments
    # For now, just send periodic heartbeat
    while True:
        try:
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            })
            await asyncio.sleep(30)
        except WebSocketDisconnect:
            break


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

