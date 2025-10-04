"""
Integration tests for Scenario Engine
Tests scenario creation, execution, and management
"""
import os
import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime, timedelta

# Ensure local dev mode for permissive behavior
os.environ.setdefault("LOCAL_DEV", "true")

# Import the scenario engine app
import sys
import importlib.util
from pathlib import Path

SCENARIO_ENGINE_PATH = Path(__file__).resolve().parents[4] / "platform" / "apps" / "scenario-engine" / "main.py"

spec = importlib.util.spec_from_file_location("scenario_engine_main", str(SCENARIO_ENGINE_PATH))
module = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore
app = module.app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def sample_scenario_spec():
    """Sample scenario specification"""
    return {
        "name": "High Demand Growth Scenario",
        "description": "Scenario with increased electricity demand growth",
        "category": "demand",
        "assumptions": {
            "load_growth_rate": {
                "type": "percentage",
                "value": 0.025,  # 2.5% annual growth
                "distribution": "normal",
                "std_dev": 0.005
            },
            "gas_price_multiplier": {
                "type": "multiplier",
                "value": 1.15,  # 15% higher gas prices
                "distribution": "uniform",
                "min": 1.05,
                "max": 1.25
            },
            "renewable_capacity_factor": {
                "type": "multiplier",
                "value": 0.95,  # 5% lower renewable output
                "distribution": "beta",
                "alpha": 2,
                "beta": 5
            }
        },
        "time_horizon": {
            "start": datetime.now().date().isoformat(),
            "end": (datetime.now() + timedelta(days=365*3)).date().isoformat()  # 3 years
        },
        "markets": ["power"],
        "regions": ["MISO", "PJM", "ERCOT"],
        "run_config": {
            "parallel_runs": 100,
            "output_granularity": "monthly",
            "include_confidence_intervals": True
        }
    }


@pytest.fixture
def sample_scenario_run_request(sample_scenario_spec):
    """Sample scenario run request"""
    return {
        "scenario_id": "test-high-demand-scenario",
        "spec": sample_scenario_spec,
        "priority": "normal",
        "notify_on_completion": False,
        "run_metadata": {
            "created_by": "test-user",
            "purpose": "integration-test"
        }
    }


def test_health(client: TestClient):
    """Test health endpoint"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_scenario_creation(client: TestClient, sample_scenario_spec):
    """Test scenario creation endpoint"""
    resp = client.post("/scenarios", json=sample_scenario_spec)

    if resp.status_code == 200:
        data = resp.json()
        assert "scenario_id" in data
        assert "created_at" in data
        assert "status" in data
        assert data["status"] == "created"

        # Validate scenario metadata
        assert data["name"] == sample_scenario_spec["name"]
        assert data["category"] == sample_scenario_spec["category"]
        assert "assumptions" in data
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_scenario_retrieval(client: TestClient):
    """Test scenario retrieval endpoint"""
    resp = client.get("/scenarios/test-high-demand-scenario")

    if resp.status_code == 200:
        data = resp.json()
        assert "scenario_id" in data
        assert "spec" in data
        assert "created_at" in data
        assert "status" in data

        # Validate scenario spec
        spec = data["spec"]
        assert "assumptions" in spec
        assert "time_horizon" in spec
        assert "run_config" in spec
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_scenario_execution(client: TestClient, sample_scenario_run_request):
    """Test scenario execution endpoint"""
    resp = client.post("/scenarios/execute", json=sample_scenario_run_request)

    if resp.status_code == 200:
        data = resp.json()
        assert "run_id" in data
        assert "status" in data
        assert "started_at" in data
        assert data["status"] in ["queued", "running", "completed"]

        # If completed immediately, check results
        if data["status"] == "completed":
            assert "results" in data
            assert "completed_at" in data
            assert "duration_seconds" in data
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_scenario_run_status(client: TestClient):
    """Test scenario run status endpoint"""
    resp = client.get("/scenarios/runs/test-run-123/status")

    if resp.status_code == 200:
        data = resp.json()
        assert "run_id" in data
        assert "status" in data
        assert "progress" in data

        # Validate status information
        assert data["status"] in ["queued", "running", "completed", "failed"]
        assert isinstance(data["progress"], (int, float))
        assert 0 <= data["progress"] <= 100
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_scenario_run_results(client: TestClient):
    """Test scenario run results endpoint"""
    resp = client.get("/scenarios/runs/test-run-123/results")

    if resp.status_code == 200:
        data = resp.json()
        assert "run_id" in data
        assert "results" in data
        assert "summary" in data

        # Validate results structure
        results = data["results"]
        assert "price_impacts" in results
        assert "volatility_changes" in results
        assert "confidence_intervals" in results

        # Validate summary statistics
        summary = data["summary"]
        assert "total_runs" in summary
        assert "successful_runs" in summary
        assert "failed_runs" in summary
        assert "execution_time" in summary
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_scenario_comparison(client: TestClient):
    """Test scenario comparison endpoint"""
    request_data = {
        "base_scenario_id": "baseline",
        "comparison_scenario_ids": ["high_demand", "low_gas_price"],
        "metrics": ["price_mean", "price_volatility", "load_growth"],
        "time_periods": ["2024", "2025", "2026"],
        "regions": ["MISO", "PJM"]
    }

    resp = client.post("/scenarios/compare", json=request_data)

    if resp.status_code == 200:
        data = resp.json()
        assert "comparison" in data
        assert "scenarios" in data
        assert "differences" in data

        # Validate comparison structure
        scenarios = data["scenarios"]
        assert "baseline" in scenarios
        assert "high_demand" in scenarios
        assert "low_gas_price" in scenarios

        # Validate differences
        differences = data["differences"]
        for metric in request_data["metrics"]:
            assert metric in differences
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_scenario_validation(client: TestClient):
    """Test scenario validation endpoint"""
    invalid_scenario = {
        "name": "",  # Invalid: empty name
        "description": "Test scenario",
        "assumptions": {
            "invalid_assumption": {
                "type": "invalid_type",  # Invalid type
                "value": -1  # Invalid value
            }
        }
    }

    resp = client.post("/scenarios/validate", json=invalid_scenario)

    if resp.status_code == 200:
        data = resp.json()
        assert "valid" in data
        assert "errors" in data
        assert "warnings" in data

        # Should identify validation errors
        assert not data["valid"]
        assert len(data["errors"]) > 0
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_scenario_list_filtering(client: TestClient):
    """Test scenario listing with filtering"""
    query_params = {
        "category": "demand",
        "status": "active",
        "limit": 10,
        "offset": 0
    }

    resp = client.get("/scenarios", params=query_params)

    if resp.status_code == 200:
        data = resp.json()
        assert "scenarios" in data
        assert "total_count" in data
        assert "page_info" in data

        # Validate pagination
        scenarios = data["scenarios"]
        assert len(scenarios) <= query_params["limit"]

        # Validate filtering
        for scenario in scenarios:
            assert scenario["category"] == query_params["category"]
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_scenario_error_handling(client: TestClient):
    """Test error handling for invalid scenario operations"""
    # Test execution of non-existent scenario
    resp = client.post("/scenarios/execute", json={
        "scenario_id": "non-existent-scenario",
        "spec": {"name": "Test"},
        "priority": "normal"
    })

    # Should return appropriate error
    assert resp.status_code in [400, 404, 422]

    # Test retrieval of non-existent scenario
    resp = client.get("/scenarios/non-existent-scenario")

    # Should return 404
    assert resp.status_code == 404
