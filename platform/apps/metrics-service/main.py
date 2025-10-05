"""
Metrics Service - Centralized metrics collection for 254Carbon platform

Collects and exports metrics from all microservices.
Provides Prometheus-compatible metrics endpoint.

Port: 8012
"""
import logging
import os
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from prometheus_exporter import (
    track_request_metric,
    track_latency_metric,
    track_connection_metric,
    get_metrics_text,
    clear_all_metrics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="254Carbon Metrics Service",
    description="Centralized metrics collection and Prometheus export",
    version="1.0.0",
)


# Models
class TrackRequest(BaseModel):
    endpoint: str
    method: str = "GET"
    status: int = 200
    service: str = "unknown"


class TrackLatency(BaseModel):
    endpoint: str
    duration_seconds: float
    service: str = "unknown"


class TrackConnection(BaseModel):
    connection_type: str
    delta: int  # +1 for new connection, -1 for closed connection
    service: str = "unknown"


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    metrics_count: int


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        metrics_count=0,  # Could count registered metrics
    )


@app.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    Scraped by Prometheus server.
    """
    metrics_text = get_metrics_text()
    return PlainTextResponse(content=metrics_text, media_type="text/plain")


@app.post("/metrics/track")
async def track_request(request: TrackRequest):
    """
    Track an API request.
    
    Records request count by endpoint, method, status, and service.
    """
    try:
        track_request_metric(
            endpoint=request.endpoint,
            method=request.method,
            status=str(request.status),
            service=request.service,
        )
        
        return {"status": "tracked"}
        
    except Exception as e:
        logger.error(f"Error tracking request: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/metrics/latency")
async def track_latency(request: TrackLatency):
    """
    Track request latency.
    
    Records latency histogram for endpoint.
    """
    try:
        track_latency_metric(
            endpoint=request.endpoint,
            duration=request.duration_seconds,
            service=request.service,
        )
        
        return {"status": "tracked"}
        
    except Exception as e:
        logger.error(f"Error tracking latency: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/metrics/connection")
async def track_connection(request: TrackConnection):
    """
    Track active connections.
    
    Updates gauge for active connections (+1 for new, -1 for closed).
    """
    try:
        track_connection_metric(
            connection_type=request.connection_type,
            delta=request.delta,
            service=request.service,
        )
        
        return {"status": "tracked"}
        
    except Exception as e:
        logger.error(f"Error tracking connection: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/metrics/clear")
async def clear_metrics():
    """
    Clear all metrics (for testing/debugging).
    
    WARNING: Use with caution in production.
    """
    try:
        clear_all_metrics()
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Error clearing metrics: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8012"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )

