"""
Integration tests for Marketplace Service.
"""
import pytest
from datetime import date, datetime, timedelta
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, db


class TestMarketplace:
    """Test suite for marketplace functionality."""

    @classmethod
    def setup_class(cls):
        """Set up test client."""
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_partner_registration(self):
        """Test partner registration flow."""
        registration_data = {
            "company_name": "Test Weather Services Inc.",
            "contact_name": "John Doe",
            "email": "john@testweather.com",
            "website": "https://testweather.com",
            "description": "Weather data provider for energy markets",
            "data_products": ["weather_forecasts", "solar_irradiance"]
        }

        response = self.client.post(
            "/api/v1/marketplace/partners/register",
            json=registration_data
        )
        assert response.status_code == 200

        data = response.json()
        assert "partner_id" in data
        assert "api_key" in data
        assert data["company_name"] == registration_data["company_name"]
        assert data["status"] == "pending"
        assert data["revenue_share_pct"] == 70.0

        # Save for later tests
        self.test_partner_id = data["partner_id"]
        self.test_api_key = data["api_key"]

    def test_list_products(self):
        """Test listing marketplace products."""
        response = self.client.get("/api/v1/marketplace/products")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

        # Should have demo products
        assert len(data) >= 3

        # Validate product structure
        product = data[0]
        assert "product_id" in product
        assert "partner_id" in product
        assert "name" in product
        assert "description" in product
        assert "category" in product
        assert "pricing_model" in product
        assert "status" in product

    def test_list_products_filtered_by_category(self):
        """Test listing products filtered by category."""
        response = self.client.get(
            "/api/v1/marketplace/products",
            params={"category": "forecasts"}
        )
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

        # All products should be in forecasts category
        for product in data:
            assert product["category"] == "forecasts"

    def test_create_product_without_auth(self):
        """Test creating product without API key (should fail)."""
        product_data = {
            "product_id": "PRD-TEST-001",
            "partner_id": "PTR-TEST",
            "name": "Test Product",
            "description": "Test data product",
            "category": "analytics",
            "pricing_model": "subscription",
            "monthly_subscription": 199.99,
            "documentation_url": "https://docs.test.com",
            "status": "pending"
        }

        response = self.client.post(
            "/api/v1/marketplace/products",
            json=product_data,
            headers={}  # No API key
        )

        # Should fail due to missing header
        assert response.status_code == 422  # Unprocessable entity

    def test_create_sandbox(self):
        """Test sandbox environment creation."""
        request_data = {
            "partner_id": "PTR-TEST",
            "product_id": "PRD-TEST-001",
            "test_data": {"sample": "data"}
        }

        response = self.client.post(
            "/api/v1/marketplace/sandbox",
            json=request_data
        )
        assert response.status_code == 200

        data = response.json()
        assert "sandbox_id" in data
        assert "api_key" in data
        assert "endpoint" in data
        assert "rate_limit" in data
        assert "expires_at" in data

        # API key should be a test key
        assert data["api_key"].startswith("sk_test_")

    def test_get_usage_stats(self):
        """Test retrieving usage statistics."""
        params = {
            "partner_id": "PTR-TEST",
            "start_date": (date.today() - timedelta(days=7)).isoformat(),
            "end_date": date.today().isoformat()
        }

        response = self.client.get(
            "/api/v1/marketplace/usage",
            params=params
        )
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

        # Should have usage data (mock or real)
        if len(data) > 0:
            usage = data[0]
            assert "usage_id" in usage
            assert "partner_id" in usage
            assert "product_id" in usage
            assert "calls_count" in usage
            assert "cost" in usage

    def test_get_revenue_split(self):
        """Test revenue split calculation."""
        params = {
            "partner_id": "PTR-TEST",
            "year": 2025,
            "month": 10
        }

        response = self.client.get(
            "/api/v1/marketplace/revenue",
            params=params
        )

        # Should fail for non-existent partner
        # Or return mock data depending on implementation
        assert response.status_code in [200, 404]

    def test_register_webhook(self):
        """Test webhook registration."""
        request_data = {
            "partner_id": "PTR-TEST",
            "webhook_url": "https://test.com/webhooks",
            "events": ["product_purchased", "usage_threshold"]
        }

        response = self.client.post(
            "/api/v1/marketplace/webhooks",
            json=request_data,
            headers={"X-Partner-Key": "test_key"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "webhook_id" in data
        assert "secret" in data
        assert data["partner_id"] == "PTR-TEST"
        assert data["status"] == "active"

    def test_marketplace_analytics(self):
        """Test marketplace-wide analytics."""
        response = self.client.get("/api/v1/marketplace/analytics")
        assert response.status_code == 200

        data = response.json()
        assert "total_partners" in data
        assert "active_products" in data
        assert "total_api_calls_month" in data
        assert "revenue_this_month" in data
        assert "top_products" in data
        assert "categories" in data

        # Validate data types
        assert isinstance(data["total_partners"], int)
        assert isinstance(data["active_products"], int)
        assert isinstance(data["revenue_this_month"], (int, float))
        assert isinstance(data["top_products"], list)
        assert isinstance(data["categories"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

