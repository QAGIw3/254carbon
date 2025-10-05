"""Pydantic-based settings loader."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Type

from pydantic_settings import BaseSettings


class ServiceSettings(BaseSettings):
    """Base settings class with common fields for services."""

    env: str = "local"
    log_level: str = "INFO"
    service_name: str = "service"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def load_settings(cls: Type[ServiceSettings] | None = None, **kwargs: Any) -> ServiceSettings:
    """Load settings once per process."""

    settings_cls = cls or ServiceSettings
    return settings_cls(**kwargs)


__all__ = ["ServiceSettings", "load_settings"]

