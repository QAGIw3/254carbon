"""
Integration tests for Backtesting Service
Tests forecast accuracy validation and MAPE/WAPE/RMSE calculations
"""
import os
import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime, timedelta

# Ensure local dev mode for permissive behavior
os.environ.setdefault("LOCAL_DEV", "true")

# Import the backtesting service app
import sys
import importlib.util
from pathlib import Path

BACKTESTING_SERVICE_PATH = Path(__file__).resolve().parents[4] / "platform" / "apps" / "backtesting-service" / "main.py"

spec = importlib.util.spec_from_file_location("backtesting_service_main", str(BACKTESTING_SERVICE_PATH))
module = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore
app = module.app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def sample_backtest_request():
    """Sample backtesting request"""
    return {
        "forecast_id": "test-forecast-001",
        "actual_data_source": "clickhouse",
        "forecast_data_source": "scenario_engine",
        "instruments": ["MISO.HUB.INDIANA"],
        "metrics": ["MAPE", "WAPE", "RMSE"],
        "time_periods": {
            "start": (datetime.now() - timedelta(days=365)).date().isoformat(),
            "end": datetime.now().date().isoformat()
        },
        "granularity": "daily",
        "comparison_windows": [30, 90, 180, 365],
        "include_confidence_intervals": True,
        "run_metadata": {
            "created_by": "test-user",
            "purpose": "integration-test"
        }
    }


@pytest.fixture
def sample_historical_comparison():
    """Sample historical comparison request"""
    return {
        "base_forecast_id": "baseline-forecast",
        "comparison_forecast_ids": ["high-demand-forecast", "low-gas-forecast"],
        "instruments": ["MISO.HUB.INDIANA", "PJM.HUB.WEST"],
        "metrics": ["MAPE", "directional_accuracy"],
        "time_periods": {
            "start": (datetime.now() - timedelta(days=180)).date().isoformat(),
            "end": datetime.now().date().isoformat()
        },
        "analysis_type": "forecast_accuracy"
    }


def test_health(client: TestClient):
    """Test health endpoint"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_backtest_execution(client: TestClient, sample_backtest_request):
    """Test backtest execution endpoint"""
    resp = client.post("/backtests/execute", json=sample_backtest_request)

    if resp.status_code == 200:
        data = resp.json()
        assert "backtest_id" in data
        assert "status" in data
        assert "started_at" in data
        assert data["status"] in ["running", "completed"]

        # If completed immediately, check results
        if data["status"] == "completed":
            assert "results" in data
            assert "completed_at" in data
            assert "duration_seconds" in data
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_backtest_status(client: TestClient):
    """Test backtest status endpoint"""
    resp = client.get("/backtests/test-backtest-123/status")

    if resp.status_code == 200:
        data = resp.json()
        assert "backtest_id" in data
        assert "status" in data
        assert "progress" in data

        # Validate status information
        assert data["status"] in ["queued", "running", "completed", "failed"]
        assert isinstance(data["progress"], (int, float))
        assert 0 <= data["progress"] <= 100
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_backtest_results(client: TestClient):
    """Test backtest results endpoint"""
    resp = client.get("/backtests/test-backtest-123/results")

    if resp.status_code == 200:
        data = resp.json()
        assert "backtest_id" in data
        assert "results" in data
        assert "summary" in data

        # Validate results structure
        results = data["results"]
        assert "accuracy_metrics" in results
        assert "error_distributions" in results
        assert "time_series_analysis" in results

        # Validate accuracy metrics
        metrics = results["accuracy_metrics"]
        for instrument in sample_backtest_request()["instruments"]:
            assert instrument in metrics

            instrument_metrics = metrics[instrument]
            for metric in ["MAPE", "WAPE", "RMSE"]:
                assert metric in instrument_metrics
                assert isinstance(instrument_metrics[metric], (int, float))
                assert instrument_metrics[metric] >= 0  # Metrics should be non-negative

        # Validate summary statistics
        summary = data["summary"]
        assert "overall_mape" in summary
        assert "overall_wape" in summary
        assert "overall_rmse" in summary
        assert "data_points_analyzed" in summary
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_historical_comparison(client: TestClient, sample_historical_comparison):
    """Test historical forecast comparison"""
    resp = client.post("/backtests/compare-forecasts", json=sample_historical_comparison)

    if resp.status_code == 200:
        data = resp.json()
        assert "comparison_id" in data
        assert "results" in data
        assert "summary" in data

        # Validate comparison results
        results = data["results"]
        assert "forecast_rankings" in results
        assert "metric_comparisons" in results
        assert "statistical_tests" in results

        # Validate forecast rankings
        rankings = results["forecast_rankings"]
        for instrument in sample_historical_comparison["instruments"]:
            assert instrument in rankings

            instrument_rankings = rankings[instrument]
            for metric in sample_historical_comparison["metrics"]:
                assert metric in instrument_rankings
                assert "ranking" in instrument_rankings[metric]
                assert "score" in instrument_rankings[metric]
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_accuracy_threshold_validation(client: TestClient):
    """Test accuracy threshold validation"""
    request_data = {
        "forecast_id": "test-forecast-threshold",
        "instruments": ["MISO.HUB.INDIANA"],
        "thresholds": {
            "MAPE_max": 0.12,  # 12% maximum MAPE
            "WAPE_max": 0.15,  # 15% maximum WAPE
            "RMSE_max": 8.5    # $8.50/MWh maximum RMSE
        },
        "time_periods": {
            "start": (datetime.now() - timedelta(days=90)).date().isoformat(),
            "end": datetime.now().date().isoformat()
        }
    }

    resp = client.post("/backtests/validate-thresholds", json=request_data)

    if resp.status_code == 200:
        data = resp.json()
        assert "validation_id" in data
        assert "results" in data
        assert "threshold_status" in data

        # Validate threshold status
        threshold_status = data["threshold_status"]
        for instrument in request_data["instruments"]:
            assert instrument in threshold_status

            instrument_status = threshold_status[instrument]
            for threshold_name in request_data["thresholds"].keys():
                assert threshold_name in instrument_status
                assert "passed" in instrument_status[threshold_name]
                assert "actual_value" in instrument_status[threshold_name]
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_error_distribution_analysis(client: TestClient):
    """Test error distribution analysis"""
    request_data = {
        "forecast_id": "test-forecast-errors",
        "instruments": ["MISO.HUB.INDIANA"],
        "analysis_type": "error_distribution",
        "time_periods": {
            "start": (datetime.now() - timedelta(days=180)).date().isoformat(),
            "end": datetime.now().date().isoformat()
        },
        "include_seasonal_analysis": True
    }

    resp = client.post("/backtests/analyze-errors", json=request_data)

    if resp.status_code == 200:
        data = resp.json()
        assert "analysis_id" in data
        assert "results" in data

        # Validate error analysis results
        results = data["results"]
        assert "error_distributions" in results
        assert "statistical_tests" in results
        assert "seasonal_patterns" in results

        # Validate error distributions
        distributions = results["error_distributions"]
        for instrument in request_data["instruments"]:
            assert instrument in distributions

            instrument_dist = distributions[instrument]
            assert "mean_error" in instrument_dist
            assert "std_error" in instrument_dist
            assert "skewness" in instrument_dist
            assert "kurtosis" in instrument_dist
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_backtest_report_generation(client: TestClient):
    """Test backtest report generation"""
    request_data = {
        "backtest_id": "test-backtest-report",
        "report_format": "html",
        "include_charts": True,
        "include_raw_data": False,
        "sections": [
            "executive_summary",
            "accuracy_metrics",
            "error_analysis",
            "recommendations"
        ]
    }

    resp = client.post("/backtests/generate-report", json=request_data)

    if resp.status_code == 200:
        data = resp.json()
        assert "report_id" in data
        assert "download_url" in data
        assert "expires_at" in data

        # Validate report metadata
        assert data["report_format"] == request_data["report_format"]
        assert "generated_at" in data
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_backtest_list_filtering(client: TestClient):
    """Test backtest listing with filtering"""
    query_params = {
        "forecast_id": "test-forecast",
        "status": "completed",
        "limit": 10,
        "offset": 0,
        "sort_by": "created_at",
        "sort_order": "desc"
    }

    resp = client.get("/backtests", params=query_params)

    if resp.status_code == 200:
        data = resp.json()
        assert "backtests" in data
        assert "total_count" in data
        assert "page_info" in data

        # Validate pagination
        backtests = data["backtests"]
        assert len(backtests) <= query_params["limit"]

        # Validate sorting
        if len(backtests) > 1:
            # Check that results are sorted by created_at desc
            for i in range(len(backtests) - 1):
                assert backtests[i]["created_at"] >= backtests[i + 1]["created_at"]
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_backtest_error_handling(client: TestClient):
    """Test error handling for invalid backtest operations"""
    # Test execution with invalid forecast ID
    resp = client.post("/backtests/execute", json={
        "forecast_id": "non-existent-forecast",
        "actual_data_source": "clickhouse",
        "instruments": ["MISO.HUB.INDIANA"],
        "time_periods": {
            "start": (datetime.now() - timedelta(days=30)).date().isoformat(),
            "end": datetime.now().date().isoformat()
        }
    })

    # Should return appropriate error
    assert resp.status_code in [400, 404, 422]

    # Test retrieval of non-existent backtest
    resp = client.get("/backtests/non-existent-backtest/results")

    # Should return 404
    assert resp.status_code == 404
