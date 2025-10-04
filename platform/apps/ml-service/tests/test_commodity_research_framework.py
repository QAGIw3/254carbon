"""Unit tests for commodity research framework analytics."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from commodity_research_framework import CommodityResearchFramework, DecompositionResult
from research_config import load_research_config, supply_demand_mapping
from research_persistence import ResearchPersistence


def _framework() -> CommodityResearchFramework:
    dummy_access = Mock()
    return CommodityResearchFramework(data_access=dummy_access, persistence=None)


def test_decomposition_recovers_trend_and_seasonality() -> None:
    np.random.seed(42)
    index = pd.date_range("2023-01-01", periods=365, freq="D")
    trend = pd.Series(np.linspace(50, 80, len(index)), index=index)
    seasonal = pd.Series(np.sin(2 * np.pi * index.dayofyear / 7), index=index)
    noise = pd.Series(np.random.normal(0, 0.2, len(index)), index=index)
    prices = trend + seasonal + noise

    framework = _framework()
    components = framework.decompose_time_series(prices, commodity_type="gas", decomposition_method="classical")

    recovered_trend = components["trend"].reindex(index)
    recovered_seasonal = components["seasonal"].reindex(index)

    assert recovered_trend.corr(trend) > 0.9
    seasonal_corr = np.corrcoef(seasonal.values, recovered_seasonal.values)[0, 1]
    assert seasonal_corr > 0.7


def test_detect_volatility_regimes_returns_labels() -> None:
    pytest.importorskip("sklearn")
    np.random.seed(0)
    low_vol = np.random.normal(0, 0.01, 200)
    high_vol = np.random.normal(0, 0.05, 200)
    returns = pd.Series(np.concatenate([low_vol, high_vol]), index=pd.date_range("2023-01-01", periods=400, freq="D"))

    framework = _framework()
    result = framework.detect_volatility_regimes(returns, n_regimes=2, method="kmeans", instrument_id="TEST")

    assert result.n_regimes == 2
    assert len(result.labels) == len(result.features)
    assert set(result.labels.unique()) == {"0", "1"}


def test_supply_demand_metrics_include_expected_keys() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = pd.Series(100 + np.linspace(0, 5, len(dates)), index=dates)
    inventory = pd.Series(1000 + np.linspace(0, 100, len(dates)), index=dates)
    production = pd.Series(500 + np.linspace(0, 10, len(dates)), index=dates)
    consumption = pd.Series(480 + np.linspace(0, 5, len(dates)), index=dates)

    framework = _framework()
    result = framework.model_supply_demand_balance(
        prices=prices,
        inventory_data=inventory,
        production_data=production,
        consumption_data=consumption,
        instrument_id="TEST",
        entity_id="REGION",
    )

    assert "inventory_cover_days" in result.metrics
    assert "supply_demand_balance" in result.metrics
    assert result.metrics["inventory_cover_days"].iloc[-1] > 0


def test_weather_impact_regression_finds_positive_temperature_coefficient() -> None:
    pytest.importorskip("statsmodels")
    np.random.seed(100)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    temperature = pd.Series(np.linspace(-5, 25, len(dates)), index=dates)
    noise = pd.Series(np.random.normal(0, 0.5, len(dates)), index=dates)
    prices = pd.Series(2.0 * temperature.values, index=dates) + noise

    framework = _framework()
    result = framework.analyze_weather_impact(
        prices=prices,
        temperature_data=temperature,
        entity_id="ZONE",
        window="60D",
        lags=[1],
        persist=False,
    )

    coef = result.coefficients.get("temperature", {}).get("coef")
    assert coef is not None
    assert 1.5 < coef < 2.5
    assert result.extreme_event_count >= 0


def test_persist_decomposition_inserts_rows() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    components = {
        "trend": pd.Series([1.0, 1.2, 1.4], index=index),
        "seasonal": pd.Series([0.1, 0.0, -0.1], index=index),
        "residual": pd.Series([0.0, 0.0, 0.0], index=index),
    }
    result = DecompositionResult(
        instrument_id="TEST",
        commodity_type="gas",
        method="stl",
        components=components,
        snapshot_date=datetime(2024, 1, 10),
        version="v-test",
    )

    client = Mock()
    persistence = ResearchPersistence(ch_client=client)
    inserted = persistence.persist_decomposition(result)

    assert inserted == 3
    assert client.execute.call_count == 1
    args, _ = client.execute.call_args
    assert "INSERT INTO ch.commodity_decomposition" in args[0]
    rows = args[1]
    assert len(rows) == 3
    assert rows[0][1] == "TEST"
    assert rows[0][2] == "stl"


def test_auto_regime_falls_back_when_markov_unavailable(monkeypatch) -> None:
    pytest.importorskip("sklearn")
    np.random.seed(0)
    returns = pd.Series(
        np.random.normal(0, 0.02, 1200),
        index=pd.date_range("2022-01-01", periods=1200, freq="D"),
    )

    monkeypatch.setattr("commodity_research_framework.MarkovRegression", None)
    monkeypatch.setattr("commodity_research_framework.GaussianHMM", None)

    framework = _framework()
    result = framework.detect_volatility_regimes(
        returns,
        n_regimes=3,
        method="auto",
        instrument_id="TEST",
    )

    assert result.method == "kmeans"
    assert result.n_regimes == 3


def test_supply_demand_config_mappings_complete() -> None:
    config = load_research_config()
    pipeline = config.get("pipelines", {}).get("supply_demand", {})
    instruments = pipeline.get("instruments", [])

    for instrument in instruments:
        mapping = supply_demand_mapping(instrument)
        assert mapping.get("entity_id"), f"Missing entity for {instrument}"
        for key in ("inventory", "production", "consumption"):
            section = mapping.get(key)
            assert section, f"Missing {key} mapping for {instrument}"
            assert section.get("entity_id"), f"Missing {key}.entity_id for {instrument}"
            assert section.get("variable"), f"Missing {key}.variable for {instrument}"
            assert section.get("unit"), f"Missing {key}.unit for {instrument}"


def test_weather_impact_raises_on_constant_temperature() -> None:
    pytest.importorskip("statsmodels")
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    temperature = pd.Series(15.0, index=dates)
    prices = pd.Series(np.linspace(10.0, 12.0, len(dates)), index=dates)

    framework = _framework()
    with pytest.raises(ValueError, match="no variance"):
        framework.analyze_weather_impact(
            prices=prices,
            temperature_data=temperature,
            entity_id="CONST",
            window="90D",
        )
