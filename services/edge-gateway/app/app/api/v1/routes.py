"""Primary API routes for Edge Gateway."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from carbon254.config import ServiceSettings, load_settings

from ..auth import verify_token
from ..cache import CacheStrategy, cache_response
from ..db import get_clickhouse_client, get_postgres_pool
from ..metrics import track_request
from ..schemas import HealthResponse
from ..services.curves.service import fetch_forward_curves
from ..services.instruments.service import fetch_instruments
from ..services.prices.service import fetch_price_ticks


router = APIRouter()


def get_settings() -> ServiceSettings:
    return load_settings(ServiceSettings, service_name="edge-gateway")


@router.get("/health", response_model=HealthResponse, summary="Gateway health check")
async def api_health(request: Request, settings: ServiceSettings = Depends(get_settings)) -> HealthResponse:
    track_request("health")
    return HealthResponse(status="healthy", service=settings.service_name)


@router.get("/instruments", summary="List instruments")
@cache_response("instruments", strategy=CacheStrategy.SEMI_STATIC)
async def list_instruments(
    request: Request,
    market: str | None = Query(default=None),
    product: str | None = Query(default=None),
    user: dict = Depends(verify_token),
):
    track_request("get_instruments")
    pool = await get_postgres_pool()
    return await fetch_instruments(pool, market=market, product=product)


@router.get("/prices/ticks", summary="Retrieve price ticks")
@cache_response("price_ticks", strategy=CacheStrategy.DYNAMIC)
async def get_price_ticks(
    request: Request,
    instrument_id: list[str] = Query(...),
    start_time: str = Query(...),
    end_time: str = Query(...),
    price_type: str = Query("mid"),
    user: dict = Depends(verify_token),
):
    track_request("get_price_ticks")
    ch_client = get_clickhouse_client()
    try:
        return await fetch_price_ticks(
            ch_client,
            instrument_ids=instrument_id,
            start_time=start_time,
            end_time=end_time,
            price_type=price_type,
            user=user,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/curves/forward", summary="Retrieve forward curves")
@cache_response("forward_curves", strategy=CacheStrategy.SEMI_STATIC)
async def get_forward_curves(
    request: Request,
    instrument_id: list[str] = Query(...),
    as_of_date: str = Query(...),
    scenario_id: str = Query("BASE"),
    user: dict = Depends(verify_token),
):
    track_request("get_forward_curves")
    ch_client = get_clickhouse_client()
    try:
        return await fetch_forward_curves(
            ch_client,
            instrument_ids=instrument_id,
            as_of_date=as_of_date,
            scenario_id=scenario_id,
            user=user,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

