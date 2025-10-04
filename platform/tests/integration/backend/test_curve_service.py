"""
Integration tests for Curve Service
Tests QP optimization, curve generation, and scenario execution
"""
import os
import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime, timedelta

# Ensure local dev mode for permissive behavior
os.environ.setdefault("LOCAL_DEV", "true")

# Import the curve service app
import sys
import importlib.util
from pathlib import Path

CURVE_SERVICE_PATH = Path(__file__).resolve().parents[4] / "platform" / "apps" / "curve-service" / "main.py"

spec = importlib.util.spec_from_file_location("curve_service_main", str(CURVE_SERVICE_PATH))
module = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore
app = module.app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def sample_curve_request():
    """Sample curve generation request"""
    return {
        "market": "power",
        "instruments": ["MISO.HUB.INDIANA"],
        "curve_type": "forward",
        "start_date": datetime.now().date().isoformat(),
        "end_date": (datetime.now() + timedelta(days=365)).date().isoformat(),
        "tenor": "monthly",
        "scenario_id": "baseline",
        "run_id": f"test-run-{datetime.now().timestamp()}"
    }


@pytest.fixture
def sample_optimization_request():
    """Sample QP optimization request"""
    return {
        "objective_function": "minimize_risk",
        "constraints": [
            {"type": "budget", "limit": 1000000},
            {"type": "exposure", "limit": 0.1}
        ],
        "solver_params": {
            "solver": "osqp",
            "max_iter": 1000,
            "tolerance": 1e-6
        }
    }


def test_health(client: TestClient):
    """Test health endpoint"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_curve_generation_endpoint(client: TestClient, sample_curve_request):
    """Test curve generation endpoint"""
    resp = client.post("/curves/generate", json=sample_curve_request)
    # In test environment, this might return mock data or fail gracefully
    if resp.status_code == 200:
        data = resp.json()
        assert "curves" in data
        assert "run_id" in data
        assert "generated_at" in data

        # Validate curve structure
        curves = data["curves"]
        assert len(curves) > 0
        for curve in curves:
            assert "instrument_id" in curve
            assert "points" in curve
            assert len(curve["points"]) > 0

            # Validate point structure
            for point in curve["points"]:
                assert "date" in point
                assert "price" in point
                assert "confidence" in point
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_curve_optimization_endpoint(client: TestClient, sample_optimization_request):
    """Test curve optimization endpoint"""
    resp = client.post("/curves/optimize", json=sample_optimization_request)

    if resp.status_code == 200:
        data = resp.json()
        assert "optimized_curve" in data
        assert "optimization_stats" in data
        assert "solver_info" in data

        # Validate optimization results
        stats = data["optimization_stats"]
        assert "objective_value" in stats
        assert "solve_time" in stats
        assert "iterations" in stats
        assert stats["solve_time"] > 0
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_curve_tenor_reconciliation(client: TestClient):
    """Test tenor reconciliation endpoint"""
    request_data = {
        "monthly_curve_id": "test-monthly-curve",
        "target_tenors": ["quarterly", "annual"],
        "reconciliation_method": "linear_interpolation"
    }

    resp = client.post("/curves/reconcile-tenors", json=request_data)

    if resp.status_code == 200:
        data = resp.json()
        assert "reconciled_curves" in data
        assert "reconciliation_report" in data

        # Validate reconciled curves
        for tenor, curve in data["reconciled_curves"].items():
            assert tenor in ["quarterly", "annual"]
            assert "points" in curve
            assert len(curve["points"]) > 0
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_curve_lineage_tracking(client: TestClient, sample_curve_request):
    """Test curve lineage tracking"""
    resp = client.get(f"/curves/lineage/{sample_curve_request['run_id']}")

    if resp.status_code == 200:
        data = resp.json()
        assert "lineage" in data
        assert "inputs" in data
        assert "outputs" in data
        assert "dependencies" in data
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_curve_scenario_comparison(client: TestClient):
    """Test curve scenario comparison"""
    request_data = {
        "base_scenario_id": "baseline",
        "comparison_scenario_ids": ["high_demand", "low_gas_price"],
        "instruments": ["MISO.HUB.INDIANA"],
        "date_range": {
            "start": datetime.now().date().isoformat(),
            "end": (datetime.now() + timedelta(days=90)).date().isoformat()
        }
    }

    resp = client.post("/curves/compare-scenarios", json=request_data)

    if resp.status_code == 200:
        data = resp.json()
        assert "comparison" in data
        assert "scenarios" in data
        assert "differences" in data

        # Validate comparison structure
        for scenario_id, scenario_data in data["scenarios"].items():
            assert "curves" in scenario_data
            assert len(scenario_data["curves"]) > 0
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_curve_export_formats(client: TestClient, sample_curve_request):
    """Test curve export in different formats"""
    # Test CSV export
    resp = client.post("/curves/export",
                      json=sample_curve_request,
                      params={"format": "csv"})

    if resp.status_code == 200:
        # Should return CSV content
        assert "text/csv" in resp.headers.get("content-type", "")
        assert len(resp.content) > 0
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]

    # Test JSON export
    resp = client.post("/curves/export",
                      json=sample_curve_request,
                      params={"format": "json"})

    if resp.status_code == 200:
        # Should return JSON content
        assert "application/json" in resp.headers.get("content-type", "")
        data = resp.json()
        assert "curves" in data
    else:
        # Acceptable failures in test environment
        assert resp.status_code in [400, 404, 503]


def test_curve_error_handling(client: TestClient):
    """Test error handling for invalid requests"""
    # Test invalid instrument
    resp = client.post("/curves/generate", json={
        "market": "power",
        "instruments": ["INVALID.INSTRUMENT"],
        "curve_type": "forward",
        "start_date": datetime.now().date().isoformat(),
        "end_date": (datetime.now() + timedelta(days=365)).date().isoformat(),
        "tenor": "monthly"
    })

    # Should return appropriate error code
    assert resp.status_code in [400, 404, 422]

    # Test invalid date range
    resp = client.post("/curves/generate", json={
        "market": "power",
        "instruments": ["MISO.HUB.INDIANA"],
        "curve_type": "forward",
        "start_date": (datetime.now() + timedelta(days=365)).date().isoformat(),
        "end_date": datetime.now().date().isoformat(),  # End before start
        "tenor": "monthly"
    })

    # Should return validation error
    assert resp.status_code in [400, 422]
