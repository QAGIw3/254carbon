"""Commodity research framework wrapper."""

from __future__ import annotations

from typing import Any


class CommodityResearchFramework:
    def __init__(self, *, data_access: Any, persistence: Any) -> None:
        self.data_access = data_access
        self.persistence = persistence

    def generate_time_series_decomposition(self, **kwargs: Any) -> Any:
        raise NotImplementedError
