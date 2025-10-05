"""Primary API routes for Edge Gateway."""

from fastapi import APIRouter, Depends

from carbon254.config import ServiceSettings, load_settings


router = APIRouter()


def get_settings() -> ServiceSettings:
    return load_settings(ServiceSettings, service_name="edge-gateway")


@router.get("/health", summary="API health check")
async def api_health(settings: ServiceSettings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}

