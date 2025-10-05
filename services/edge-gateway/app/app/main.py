"""Edge Gateway entrypoint."""

from fastapi import Depends, FastAPI

from carbon254.config import load_settings, ServiceSettings
from carbon254.logging import configure_json_logging

from .api.v1 import router as api_router


def get_settings() -> ServiceSettings:
    return load_settings(ServiceSettings, service_name="edge-gateway")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_json_logging(settings.log_level)

    app = FastAPI(title="Edge Gateway", version="0.0.1")
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()


@app.get("/health")
async def health(settings: ServiceSettings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}

