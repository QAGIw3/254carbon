"""Analytics Platform entrypoint."""

from fastapi import FastAPI

from carbon254.config import ServiceSettings, load_settings
from carbon254.logging import configure_json_logging

from .routers import legacy_router, v1_router


def get_settings() -> ServiceSettings:
    return load_settings(ServiceSettings, service_name="analytics-platform")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_json_logging(settings.log_level)

    app = FastAPI(title="Analytics Platform", version="0.0.1")
    app.include_router(v1_router, prefix="/api/v1")
    app.include_router(legacy_router, prefix="/legacy")
    return app


app = create_app()

