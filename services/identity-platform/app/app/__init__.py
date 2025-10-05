"""Identity Platform application package."""

from fastapi import FastAPI

from carbon254.config import ServiceSettings, load_settings
from carbon254.logging import configure_json_logging


def get_settings() -> ServiceSettings:
    return load_settings(ServiceSettings, service_name="identity-platform")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_json_logging(settings.log_level)

    app = FastAPI(title="Identity Platform", version="0.0.1")
    return app


app = create_app()

