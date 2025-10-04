from __future__ import annotations

import enum
import importlib
import sys
import types
from dataclasses import dataclass as _dc_dataclass
from datetime import date, datetime
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

stdlib_platform = importlib.import_module("platform")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "platform" / "apps"))

platform_path = str(ROOT / "platform")
platform_pkg = types.ModuleType("platform")
platform_pkg.__dict__.update(stdlib_platform.__dict__)
platform_pkg.__path__ = [platform_path]
platform_pkg.__spec__ = ModuleSpec("platform", loader=None, is_package=True)
platform_pkg.__spec__.submodule_search_locations = [platform_path]
sys.modules["platform"] = platform_pkg

apps_pkg = types.ModuleType("platform.apps")
apps_pkg.__path__ = [str(ROOT / "platform" / "apps")]
apps_pkg.__spec__ = ModuleSpec("platform.apps", loader=None, is_package=True)
apps_pkg.__spec__.submodule_search_locations = apps_pkg.__path__
sys.modules["platform.apps"] = apps_pkg

graphql_pkg = types.ModuleType("platform.apps.graphql_gateway")
graphql_pkg.__path__ = [str(ROOT / "platform" / "apps" / "graphql-gateway")]
graphql_pkg.__spec__ = ModuleSpec(
    "platform.apps.graphql_gateway", loader=None, is_package=True
)
graphql_pkg.__spec__.submodule_search_locations = graphql_pkg.__path__
sys.modules["platform.apps.graphql_gateway"] = graphql_pkg

strawberry_stub = types.ModuleType("strawberry")


def _dataclass_decorator(target=None, **kwargs):  # pragma: no cover - helper
    def wrap(cls):
        return _dc_dataclass(cls)

    if target is None:
        return wrap
    return wrap(target)


def _identity(func=None, **kwargs):  # pragma: no cover - helper
    return func


def _enum_factory(target=None, **kwargs):  # pragma: no cover - helper
    def wrap(enum_cls):
        members = {name: value for name, value in enum_cls.__dict__.items() if name.isupper()}
        return enum.Enum(enum_cls.__name__, members)

    if target is None:
        return wrap
    return wrap(target)


strawberry_stub.type = _dataclass_decorator
strawberry_stub.input = _dataclass_decorator
strawberry_stub.enum = _enum_factory
strawberry_stub.field = _identity
strawberry_scalars_stub = types.ModuleType("strawberry.scalars")
strawberry_scalars_stub.JSON = dict
sys.modules["strawberry.scalars"] = strawberry_scalars_stub

strawberry_types_stub = types.ModuleType("strawberry.types")
strawberry_types_stub.Info = object
sys.modules["strawberry.types"] = strawberry_types_stub

strawberry_stub.scalars = strawberry_scalars_stub
strawberry_stub.types = strawberry_types_stub
sys.modules["strawberry"] = strawberry_stub

from platform.apps.graphql_gateway.analytics_schema import (
    AnalyticsQuery,
    ResearchQueryId,
    ResearchQueryInput,
)
from platform.shared import cache_utils


class _NullCacheManager:
    def get(self, namespace: str, key: str) -> Optional[Any]:  # pragma: no cover - interface stub
        return None

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:  # pragma: no cover - interface stub
        return False


@pytest.fixture(autouse=True)
def disable_cache(monkeypatch):
    """Ensure cache lookups stay local during unit tests."""
    monkeypatch.setattr(cache_utils, "_global_cache_manager", _NullCacheManager())


class FakeClickHouseClient:
    def __init__(self, responses: Sequence[Tuple[List[Tuple[Any, ...]], List[str]]]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs):
        if not self._responses:
            raise AssertionError("No fake response available for query")
        parameters = parameters or {}
        self.calls.append(
            {
                "query": " ".join(query.split()),
                "params": dict(parameters),
                "settings": kwargs.get("settings"),
            }
        )
        data, columns = self._responses.pop(0)
        column_types = [(name, "String") for name in columns]
        return data, column_types


def _info_with_client(client: FakeClickHouseClient):
    return SimpleNamespace(context={"ch_client": client})


def test_correlation_pairs_filters_and_conversion():
    responses = [
        (
            [
                (date(2024, 1, 1), "NG1", "NG2", 0.92, 64),
                (date(2024, 1, 2), "NG1", "NG3", 0.85, 87),
            ],
            ["date", "instrument1", "instrument2", "correlation", "sample_count"],
        )
    ]
    client = FakeClickHouseClient(responses)
    query = AnalyticsQuery()

    result = query.correlation_pairs(
        _info_with_client(client),
        instruments=["ng1", "NG2", "ng3"],
        min_samples=50,
        limit=25,
    )

    assert len(result) == 2
    assert result[0].instrument1 == "NG1"
    assert result[0].instrument2 == "NG2"

    executed = client.calls[0]
    assert executed["params"]["limit"] == 25
    assert executed["params"]["min_samples"] == 50
    assert set(executed["params"]["instrument_filter"]) == {"NG1", "NG2", "NG3"}
    assert "ch.commodity_correlations" in executed["query"]


def test_correlation_matrix_pivots_symmetrically():
    responses = [
        ([(date(2024, 1, 3),)], ["max(date)"]),
        (
            [
                (date(2024, 1, 3), "NG1", "NG1", 1.0),
                (date(2024, 1, 3), "NG1", "NG2", 0.8),
                (date(2024, 1, 3), "NG2", "NG1", 0.8),
                (date(2024, 1, 3), "NG2", "NG2", 1.0),
            ],
            ["date", "instrument1", "instrument2", "correlation"],
        ),
    ]
    client = FakeClickHouseClient(responses)
    matrix = AnalyticsQuery().correlation_matrix(
        _info_with_client(client),
        instruments=["NG2", "ng1"],
    )

    assert matrix is not None
    assert matrix.date == date(2024, 1, 3)
    assert matrix.coefficients["NG1"]["NG2"] == pytest.approx(0.8)
    assert matrix.coefficients["NG2"]["NG1"] == pytest.approx(0.8)
    assert matrix.coefficients["NG1"]["NG1"] == pytest.approx(1.0)
    assert matrix.coefficients["NG2"]["NG2"] == pytest.approx(1.0)
    assert len(client.calls) == 2


def test_volatility_surface_requires_instrument():
    query = AnalyticsQuery()
    with pytest.raises(ValueError):
        query.volatility_surface(_info_with_client(FakeClickHouseClient([])))


def test_seasonality_decomposition_latest_returns_point():
    responses = [
        (
            [
                (date(2024, 2, 1), "NG1", "stl", 1.2, 0.5, -0.1),
            ],
            ["snapshot_date", "instrument_id", "method", "trend", "seasonal", "residual"],
        )
    ]
    client = FakeClickHouseClient(responses)
    result = AnalyticsQuery().seasonality_decomposition_latest(
        _info_with_client(client),
        instrument_id="ng1",
        method="STL",
    )

    assert result is not None
    assert result.snapshot_date == date(2024, 2, 1)
    assert result.trend == pytest.approx(1.2)


def test_research_query_runs_template_and_applies_settings():
    responses = [
        (
            [
                ("nb-1", "Alpha", "analyst", "draft", datetime(2024, 1, 1), None),
            ],
            [
                "notebook_id",
                "title",
                "author",
                "status",
                "created_at",
                "executed_at",
            ],
        )
    ]
    client = FakeClickHouseClient(responses)
    query = AnalyticsQuery()
    result = query.research_query(
        _info_with_client(client),
        ResearchQueryInput(
            query_id=ResearchQueryId.LIST_NOTEBOOKS,
            params={"status": "draft"},
            limit=10,
        ),
    )

    assert len(result) == 1
    assert result[0].columns[0] == "notebook_id"
    assert result[0].values[0] == "nb-1"

    executed = client.calls[0]
    assert executed["params"]["status"] == "draft"
    assert executed["params"]["limit"] == 10
    assert executed["settings"]["readonly"] == 1
    assert executed["settings"]["max_result_rows"] == 10
    assert "ch.research_notebooks" in executed["query"]


def test_research_query_rejects_non_mapping_params():
    query = AnalyticsQuery()
    with pytest.raises(ValueError):
        query.research_query(
            _info_with_client(FakeClickHouseClient([])),
            ResearchQueryInput(query_id=ResearchQueryId.LIST_NOTEBOOKS, params=["invalid"]),
        )
