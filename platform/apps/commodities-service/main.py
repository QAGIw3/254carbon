"""
Commodities Service
-------------------

Unified FastAPI application providing commodity market data endpoints
across gas, oil, coal, biofuels, and battery materials.

Key design points
- Composition: Each commodity group implements its own router and models.
- Data reads: Prefer ClickHouse via small query helpers; fall back to
  deterministic synthetic data for developer parity when CH is unreachable.
- Caching: Shared Redis cache utilities (thin wrapper around shared cache).
- Auth: Minimal role checks (via dependency) in each router. LOCAL_DEV bypass
  available for rapid local iteration.
- Observability: Simple /health, /ready endpoints and request-id logging.

Note: This module intentionally keeps orchestration minimal and mounts the
routers; the heavy lifting lives in `deps/*` and `*/router.py` modules.
"""
import asyncio
import logging
import os
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deps.cache import get_cache_manager
from deps.db import ping_clickhouse, ping_postgres
from gas.router import router as gas_router
from oil.router import router as oil_router
from coal.router import router as coal_router
from biofuels.router import router as biofuels_router
from battery_materials.router import router as battery_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("commodities-service")

SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")

app = FastAPI(
    title="Commodities Service",
    description="Unified commodity market data surface",
    version=SERVICE_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):  # type: ignore[override]
    """
    Attach a request ID and log structured entries.

    - Ensures every request/response carries an `x-request-id` header for easy
      correlation in logs and upstream proxies.
    - Keeps the middleware intentionally lightweight to minimize overhead.
    """

    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    logger.info("%s %s request_id=%s", request.method, request.url.path, request_id)

    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled error processing request %s", request_id)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"service": "commodities", "version": SERVICE_VERSION}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/ready")
async def readiness() -> Dict[str, Any]:
    """
    Report readiness by pinging backing dependencies.

    - ClickHouse and Postgres checks use quick health queries/pings.
    - Redis check piggybacks on the cache manager to avoid duplicate clients.
    """
    ch_task = asyncio.create_task(ping_clickhouse())
    pg_task = asyncio.create_task(ping_postgres())

    cache_manager = get_cache_manager()
    loop = asyncio.get_running_loop()

    async def _ping_redis() -> bool:
        client = getattr(cache_manager, "client", None)
        if client is None:
            return False
        try:
            return await loop.run_in_executor(None, client.ping)
        except Exception:
            return False

    redis_task = asyncio.create_task(_ping_redis())

    ch_ready, pg_ready, redis_ready = await asyncio.gather(ch_task, pg_task, redis_task)

    status = {"clickhouse": ch_ready, "postgres": pg_ready, "redis": redis_ready}
    if all(status.values()):
        return {"status": "ready", **status}

    raise HTTPException(status_code=503, detail=status)


app.include_router(gas_router)
app.include_router(oil_router)
app.include_router(coal_router)
app.include_router(biofuels_router)
app.include_router(battery_router)


__all__ = ["app"]
