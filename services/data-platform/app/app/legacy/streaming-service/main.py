"""
Streaming Service - Real-time data streaming for 254Carbon platform

Responsibilities:
- WebSocket connections for real-time price streaming
- Server-Sent Events (SSE) for HTTP-based streaming
- Connection management and lifecycle
- Kafka consumer integration
- Real-time price fanout

Port: 8001
"""
import asyncio
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stream_manager import StreamManager
from clients.auth_client import verify_ws_token
from clients.entitlements_client import check_entitlement
from clients.metrics_client import track_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stream manager for connections
stream_manager = StreamManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application."""
    logger.info("Starting Streaming Service...")
    
    logger.info("Streaming Service started successfully")
    yield
    
    logger.info("Shutting down Streaming Service...")
    await stream_manager.shutdown()


# Create FastAPI application
app = FastAPI(
    title="254Carbon Streaming Service",
    description="Real-time data streaming via WebSocket and SSE",
    version="1.0.0",
    lifespan=lifespan,
)

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
    active_websockets: int
    active_sse: int


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        active_websockets=len(stream_manager.connections),
        active_sse=len(stream_manager.all_queues) + sum(
            len(qs) for qs in stream_manager.instrument_queues.values()
        ),
    )


# WebSocket streaming endpoint
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time price streaming via WebSocket."""
    await websocket.accept()

    instrument_ids = []

    try:
        # Authenticate
        auth_msg = await websocket.receive_json()

        if auth_msg.get("type") == "subscribe":
            instrument_ids = auth_msg.get("instruments", [])
            commodity_types = auth_msg.get("commodities", [])
            subscribe_all = bool(auth_msg.get("all", False))
            api_key = auth_msg.get("api_key")

            # Production: validate JWT; Dev: allow dev-key
            if os.getenv("LOCAL_DEV", "true") != "true":
                try:
                    user_claims = await verify_ws_token(api_key)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Unauthorized"})
                    await websocket.close()
                    return

                # Check entitlements
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
            await stream_manager.register(
                websocket,
                instrument_ids,
                commodity_types=commodity_types,
                subscribe_all=subscribe_all
            )

            # Track metrics
            await track_request("/ws/stream", "WEBSOCKET", 101)

            # Send confirmation
            await websocket.send_json({
                "type": "subscribed",
                "instruments": instrument_ids,
                "commodities": commodity_types,
                "all": subscribe_all,
                "message": f"Subscribed to {len(instrument_ids)} instruments"
            })

            # Start streaming
            if os.getenv("LOCAL_DEV", "true") == "true":
                await stream_mock_data(websocket, instrument_ids)
            else:
                # In production, Kafka consumer in stream_manager handles data
                # Keep connection alive
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


async def stream_mock_data(websocket: WebSocket, instrument_ids: List[str]):
    """Stream mock price data for local development."""
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
                price = max(0, price)

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


async def stream_kafka_data(websocket: WebSocket, instrument_ids: List[str]):
    """Stream real price data from Kafka (keep alive)."""
    # In production, Kafka consumer in stream_manager handles data
    # Just keep connection alive with periodic heartbeats
    while True:
        try:
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            })
            await asyncio.sleep(30)
        except WebSocketDisconnect:
            break


# Server-Sent Events endpoint
@app.get("/sse/stream")
async def sse_stream(
    request: Request,
    instruments: List[str] = Query(default=[]),
    commodities: List[str] = Query(default=[]),
    all: bool = Query(default=False, description="Subscribe to all updates"),
    api_key: Optional[str] = Query(default=None, description="API key for authentication"),
):
    """
    HTTP SSE endpoint for real-time price updates.
    
    Emits event-stream data frames with periodic keep-alives.
    """
    # Authentication (optional in dev, required in prod)
    if os.getenv("LOCAL_DEV", "true") != "true":
        if not api_key:
            return StreamingResponse(
                iter([b"event: error\ndata: {\"message\": \"API key required\"}\n\n"]),
                media_type="text/event-stream"
            )
        
        try:
            user_claims = await verify_ws_token(api_key)
            
            # Check entitlements
            for inst_id in instruments:
                entitled = await check_entitlement(user_claims, inst_id, "stream")
                if not entitled:
                    return StreamingResponse(
                        iter([f"event: error\ndata: {{\"message\": \"Not entitled to {inst_id}\"}}\n\n".encode()]),
                        media_type="text/event-stream"
                    )
        except Exception as e:
            logger.error(f"SSE authentication failed: {e}")
            return StreamingResponse(
                iter([b"event: error\ndata: {\"message\": \"Authentication failed\"}\n\n"]),
                media_type="text/event-stream"
            )

    # Track metrics
    await track_request("/sse/stream", "GET", 200)

    # Create queue for this SSE connection
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    await stream_manager.register_http(queue, instruments, commodity_types=commodities, subscribe_all=all)

    async def event_generator():
        try:
            # Initial ack
            yield "event: subscribed\n".encode("utf-8")
            payload = json.dumps({"instruments": instruments, "commodities": commodities, "all": all})
            yield f"data: {payload}\n\n".encode("utf-8")

            keepalive_interval = 15
            last_sent = asyncio.get_event_loop().time()

            while True:
                # Heartbeat keep-alive
                now = asyncio.get_event_loop().time()
                if now - last_sent >= keepalive_interval:
                    yield b":ka\n\n"
                    last_sent = now

                try:
                    item = await asyncio.wait_for(queue.get(), timeout=keepalive_interval)
                    data = json.dumps(item, default=str)
                    yield f"data: {data}\n\n".encode("utf-8")
                    last_sent = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    continue

                if await request.is_disconnected():
                    break
        finally:
            await stream_manager.unregister_http(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )

