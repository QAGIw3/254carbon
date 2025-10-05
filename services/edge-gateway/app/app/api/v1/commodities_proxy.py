"""
Commodity endpoints proxy for the API Gateway.

This router forwards `/api/v1/commodities/*` requests to the consolidated
commodities-service (default `http://commodities-service:8012`).

Notes
- Authentication is still enforced at the gateway via `verify_token`.
- Headers and query params are forwarded; hop-by-hop headers are filtered.
- Keeps the gateway API surface backward-compatible while services consolidate.
"""
import os
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from auth import verify_token

# Base URL for the commodities-service; can be overridden at deploy time to
# support different DNS or port wiring in various environments.
COMMODITIES_SERVICE_URL = os.getenv(
    "COMMODITIES_SERVICE_URL",
    "http://commodities-service:8012",
)

router = APIRouter(prefix="/api/v1/commodities", tags=["commodities-service"])


async def _forward(path: str, request: Request) -> Response:
    """Forward the incoming request to the commodities-service.

    Preserves method, query string, and request body. Returns a raw Response
    with upstream body and status code, filtering hop-by-hop headers.
    """
    url = f"{COMMODITIES_SERVICE_URL}/api/v1/commodities/{path}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            upstream_response = await client.request(
                method=request.method,
                url=url,
                params=request.query_params,
                content=await request.body(),
                headers=headers,
            )
    except httpx.RequestError as exc:  # pragma: no cover - network failure
        raise HTTPException(status_code=502, detail=f"Commodities service unreachable: {exc}") from exc

    response = Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type"),
    )

    hop_by_hop = {"content-length", "content-encoding", "transfer-encoding"}
    for key, value in upstream_response.headers.items():
        if key.lower() in hop_by_hop:
            continue
        response.headers[key] = value

    return response


@router.get("/gas/prices")
async def proxy_gas_prices(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("gas/prices", request)


@router.get("/gas/storage")
async def proxy_gas_storage(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("gas/storage", request)


@router.get("/gas/pipelines")
async def proxy_gas_pipelines(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("gas/pipelines", request)


@router.get("/gas/lng")
async def proxy_gas_lng(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("gas/lng", request)


@router.get("/oil/curves")
async def proxy_oil_curves(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("oil/curves", request)


@router.get("/coal/indices")
async def proxy_coal_indices(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("coal/indices", request)


@router.get("/coal/stockpiles")
async def proxy_coal_stockpiles(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("coal/stockpiles", request)


@router.get("/biofuels/rin-prices")
async def proxy_rin_prices(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("biofuels/rin-prices", request)


@router.get("/battery-materials/lithium")
async def proxy_lithium(request: Request, _: dict = Depends(verify_token)) -> Response:
    return await _forward("battery-materials/lithium", request)


__all__ = ["router"]
