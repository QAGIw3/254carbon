import importlib
import types
from importlib.machinery import ModuleSpec
from datetime import date
from pathlib import Path
from typing import List
import sys

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

data_pkg = types.ModuleType("platform.data")
data_pkg.__path__ = [str(ROOT / "platform" / "data")]
data_pkg.__spec__ = ModuleSpec("platform.data", loader=None, is_package=True)
data_pkg.__spec__.submodule_search_locations = data_pkg.__path__
sys.modules["platform.data"] = data_pkg

class _FakeKafkaProducer:  # pragma: no cover - utility stub
    def __init__(self, *_, **__):
        pass

    def send(self, *_, **__):
        return None

    def flush(self):
        return None


kafka_stub = types.ModuleType("kafka")
kafka_stub.KafkaProducer = _FakeKafkaProducer
sys.modules["kafka"] = kafka_stub

kafka_producer_stub = types.ModuleType("kafka.producer")
kafka_producer_stub.KafkaProducer = _FakeKafkaProducer
sys.modules["kafka.producer"] = kafka_producer_stub

import numpy as np
import pandas as pd
import pytest

from gas_coal_analytics.analytics import storage_arbitrage as sa
from gas_coal_analytics.analytics import weather_impact as wi
from gas_coal_analytics.analytics import gas_basis as gb
from gas_coal_analytics.analytics import coal_to_gas as ctg
from gas_coal_analytics.clients import clickhouse
from gas_coal_analytics.jobs import weather_metrics_job as wm_job


@pytest.fixture(autouse=True)
def reset_clickhouse(monkeypatch):
    captured: List = []

    def fake_insert(table: str, rows: List[dict]):
        captured.append((table, rows))

    monkeypatch.setattr(clickhouse, "insert_rows", fake_insert)
    monkeypatch.setattr(sa, "insert_rows", fake_insert, raising=False)
    monkeypatch.setattr(wi, "insert_rows", fake_insert, raising=False)
    monkeypatch.setattr(gb, "insert_rows", fake_insert, raising=False)
    monkeypatch.setattr(ctg, "insert_rows", fake_insert, raising=False)
    monkeypatch.setattr(wm_job, "insert_rows", fake_insert, raising=False)
    return captured


def test_storage_arbitrage_schedule_generates_actions(monkeypatch, reset_clickhouse):
    def fake_query(sql: str, params=None):
        if "forward_curve_points" in sql:
            days = pd.date_range(date(2024, 1, 1), periods=120, freq="D")
            prices = pd.Series(3.0 + 0.5 * np.sin(np.linspace(0, 4, len(days))), index=days)
            return pd.DataFrame({"date": prices.index.date, "price": prices.values})
        if "supply_demand_metrics" in sql:
            return pd.DataFrame({"metric_value": [55.0]})
        raise AssertionError("Unexpected query")

    monkeypatch.setattr(sa, "query_dataframe", fake_query)

    calc = sa.StorageArbitrageCalculator()
    result = calc.compute("HENRY", date(2024, 1, 1))
    assert result.schedule, "Expected non-empty schedule"
    assert any(entry.action != "HOLD" for entry in result.schedule)


def test_weather_impact_analyzer_returns_coefficients(monkeypatch, reset_clickhouse):
    def fake_query(sql: str, params=None):
        if "market_price_daily_agg" in sql:
            days = pd.date_range(date(2023, 7, 1), periods=200, freq="D")
            prices = pd.Series(2.5 + 0.1 * np.sin(np.linspace(0, 10, len(days))), index=days)
            return pd.DataFrame({"date": prices.index.date, "price": prices.values})
        if "supply_demand_metrics" in sql:
            days = pd.date_range(date(2023, 7, 1), periods=200, freq="D")
            hdd = pd.Series(np.maximum(20 - np.sin(np.linspace(0, 5, len(days))) * 10, 0), index=days)
            cdd = pd.Series(np.maximum(np.sin(np.linspace(0, 5, len(days))) * 10 - 5, 0), index=days)
            records = []
            for ts, val in hdd.items():
                records.append({"date": ts.date(), "metric_name": "hdd", "metric_value": val})
            for ts, val in cdd.items():
                records.append({"date": ts.date(), "metric_name": "cdd", "metric_value": val})
            return pd.DataFrame(records)
        raise AssertionError("Unexpected query")

    monkeypatch.setattr(wi, "query_dataframe", fake_query)

    analyzer = wi.WeatherImpactAnalyzer(window=90)
    coeffs = analyzer.run("HENRY", date(2024, 1, 1))
    assert {c.coef_type for c in coeffs} == {"hdd", "cdd"}
    assert all(abs(c.coefficient) < 10 for c in coeffs)


def test_coal_to_gas_switching_bounds(monkeypatch, reset_clickhouse):
    def fake_price(self, instrument_id: str, as_of: date):
        if "DOMINION" in instrument_id:
            return 3.0
        if "API" in instrument_id:
            return 60.0
        return 2.5

    monkeypatch.setattr(ctg.CoalToGasSwitchingCalculator, "_load_price", fake_price)

    calc = ctg.CoalToGasSwitchingCalculator(co2_price_default=20.0)
    result = calc.compute("PJM", date(2024, 1, 1))
    assert 0 <= result.switch_share <= 1
    assert result.coal_cost_mwh > 0


def test_gas_basis_modeler_predicts(monkeypatch, reset_clickhouse):
    def fake_query(sql: str, params=None):
        if "market_price_daily_agg" in sql:
            days = pd.date_range(date(2023, 6, 1), periods=200, freq="D")
            base = 2.8 + 0.15 * np.sin(np.linspace(0, 8, len(days)))
            df = pd.DataFrame({"date": days.date, "avg_price": base})
            if params and params.get("hub") == "HENRY":
                return df
            if params and params.get("hub"):
                df["avg_price"] = base + 0.2
                return df
        if "supply_demand_metrics" in sql:
            days = pd.date_range(date(2023, 6, 1), periods=200, freq="D")
            data = []
            for metric in gb.FEATURE_METRICS:
                data.extend({"date": d.date(), "metric_name": metric, "metric_value": 0.5} for d in days)
            return pd.DataFrame(data)
        raise AssertionError("Unexpected query")

    monkeypatch.setattr(gb, "query_dataframe", fake_query)

    modeler = gb.GasBasisModeler(lookback_days=120)
    result = modeler.compute("DAWN", date(2024, 1, 1))
    assert result.predicted_basis is not None
    assert abs(result.predicted_basis) < 5

def test_hdd_cdd_job_uses_weather_connectors(monkeypatch, reset_clickhouse):
    class DummyNOAA:
        def __init__(self, config):
            self.config = config

        def pull_or_subscribe(self):
            return iter([
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "variable": "temp_c",
                    "value": 5.0,
                    "unit": "C",
                }
            ])

    class DummyNASA:
        def __init__(self, config):
            self.config = config

        def pull_or_subscribe(self):
            return iter([])

    monkeypatch.setattr(wm_job, "NOAACDOConnector", DummyNOAA)
    monkeypatch.setattr(wm_job, "NASAPowerConnector", DummyNASA)

    rows = wm_job.run_hdd_cdd_metrics_job(date(2024, 1, 1), ["PJM"])
    assert len(rows) == 2
    hdd_row = next(r for r in rows if r["metric_name"] == "hdd")
    cdd_row = next(r for r in rows if r["metric_name"] == "cdd")
    assert hdd_row["metric_value"] > 0
    assert cdd_row["metric_value"] == 0
