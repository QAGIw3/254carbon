"""
Integration tests for LMP Decomposition Service.
"""
import pytest
import asyncio
from datetime import datetime, date, timedelta
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


class TestLMPDecomposition:
    """Test suite for LMP decomposition functionality."""

    @classmethod
    def setup_class(cls):
        """Set up test client."""
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_lmp_decompose_single_node(self):
        """Test LMP decomposition for a single node."""
        request_data = {
            "node_ids": ["PJM.HUB.WEST"],
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "iso": "PJM"
        }

        response = self.client.post("/api/v1/lmp/decompose", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Validate structure
        component = data[0]
        assert "timestamp" in component
        assert "node_id" in component
        assert "lmp_total" in component
        assert "energy_component" in component
        assert "congestion_component" in component
        assert "loss_component" in component

        # Validate decomposition sums correctly
        lmp_total = component["lmp_total"]
        component_sum = (
            component["energy_component"] +
            component["congestion_component"] +
            component["loss_component"]
        )

        # Allow small floating point tolerance
        assert abs(lmp_total - component_sum) < 0.1

    def test_lmp_decompose_multiple_nodes(self):
        """Test LMP decomposition for multiple nodes."""
        request_data = {
            "node_ids": ["PJM.HUB.WEST", "PJM.WESTERN", "PJM.EASTERN"],
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=2)).isoformat(),
            "iso": "PJM"
        }

        response = self.client.post("/api/v1/lmp/decompose", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

        # Should have at least 2 hours * 3 nodes = 6 observations
        assert len(data) >= 6

        # Validate node IDs are present
        node_ids = {item["node_id"] for item in data}
        assert "PJM.HUB.WEST" in node_ids
        assert "PJM.WESTERN" in node_ids
        assert "PJM.EASTERN" in node_ids

    def test_ptdf_calculation(self):
        """Test PTDF calculation."""
        request_data = {
            "source_node": "PJM.HUB.WEST",
            "sink_node": "PJM.EASTERN",
            "constraint_id": "WEST_TO_EAST",
            "iso": "PJM"
        }

        response = self.client.post("/api/v1/lmp/ptdf", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "ptdf_value" in data
        assert "source_node" in data
        assert "sink_node" in data
        assert "interpretation" in data

        # PTDF should be between -1 and 1
        ptdf_value = data["ptdf_value"]
        assert -1.0 <= ptdf_value <= 1.0

    def test_basis_surface_calculation(self):
        """Test hub-to-node basis surface calculation."""
        request_data = {
            "hub_id": "PJM.HUB.WEST",
            "node_ids": ["PJM.WESTERN", "PJM.EASTERN"],
            "as_of_date": date.today().isoformat(),
            "iso": "PJM"
        }

        response = self.client.post("/api/v1/lmp/basis-surface", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "hub_id" in data
        assert "basis_surface" in data

        basis_surface = data["basis_surface"]
        assert isinstance(basis_surface, list)
        assert len(basis_surface) == 2  # Two nodes requested

        # Validate basis statistics
        for basis in basis_surface:
            assert "node_id" in basis
            assert "mean_basis" in basis
            assert "std_basis" in basis
            assert "correlation_to_hub" in basis

            # Correlation should be between -1 and 1
            assert -1.0 <= basis["correlation_to_hub"] <= 1.0

    def test_congestion_forecast(self):
        """Test congestion forecasting."""
        params = {
            "node_id": "PJM.HUB.WEST",
            "forecast_date": date.today().isoformat(),
            "iso": "PJM"
        }

        response = self.client.get("/api/v1/lmp/congestion-forecast", params=params)
        assert response.status_code == 200

        data = response.json()
        assert "node_id" in data
        assert "forecast_date" in data
        assert "forecasted_congestion" in data
        assert "binding_constraints" in data

        forecasts = data["forecasted_congestion"]
        assert isinstance(forecasts, list)
        assert len(forecasts) == 24  # 24 hours

        # Validate forecast structure
        for forecast in forecasts:
            assert "timestamp" in forecast
            assert "congestion_forecast" in forecast
            assert "confidence" in forecast

    def test_ptdf_matrix_visualization(self):
        """Test PTDF matrix visualization endpoint."""
        params = {
            "source_nodes": ["PJM.HUB.WEST", "PJM.WESTERN"],
            "sink_nodes": ["PJM.EASTERN"],
            "constraints": ["WEST_TO_EAST"],
            "iso": "PJM"
        }

        response = self.client.get("/api/v1/lmp/visualization/ptdf-matrix", params=params)
        assert response.status_code == 200

        data = response.json()
        assert "ptdf_matrix" in data
        assert "matrix_shape" in data

        matrix_shape = data["matrix_shape"]
        assert matrix_shape[0] == 2  # 2 source nodes
        assert matrix_shape[1] == 1  # 1 sink node
        assert matrix_shape[2] == 1  # 1 constraint

    def test_basis_heatmap_data(self):
        """Test basis heatmap visualization endpoint."""
        params = {
            "hub_id": "PJM.HUB.WEST",
            "node_ids": ["PJM.WESTERN", "PJM.EASTERN", "PJM.CENTRAL"],
            "as_of_date": date.today().isoformat(),
            "iso": "PJM"
        }

        response = self.client.get("/api/v1/lmp/visualization/basis-heatmap", params=params)
        assert response.status_code == 200

        data = response.json()
        assert "hub_id" in data
        assert "basis_data" in data
        assert "node_count" in data

        basis_data = data["basis_data"]
        assert len(basis_data) == 3  # 3 nodes requested

        # Validate all nodes have basis statistics
        for node_basis in basis_data:
            assert "node_id" in node_basis
            assert "mean_basis" in node_basis
            assert "std_basis" in node_basis
            assert "correlation" in node_basis
            assert "volatility_ratio" in node_basis

    def test_invalid_node_id(self):
        """Test error handling for invalid node IDs."""
        request_data = {
            "node_ids": [],  # Empty list
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "iso": "PJM"
        }

        response = self.client.post("/api/v1/lmp/decompose", json=request_data)
        # Should handle gracefully with empty list
        assert response.status_code in [200, 400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

