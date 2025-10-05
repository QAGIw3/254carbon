"""
Smoke tests for the commodities-service routers.

These tests exercise the happy-path behavior of service endpoints under
LOCAL_DEV bypass. They purposefully do not require external dependencies
and serve as a safety net to detect import/regression issues.
"""

import os
from datetime import date

os.environ.setdefault("LOCAL_DEV", "true")  # Enable dev-bypass roles

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_gas_prices_fallback():
    today = date.today()
    params = {
        "instrument_id": "NG_HENRY_HUB",
        "start_date": today.isoformat(),
        "end_date": today.isoformat(),
    }
    response = client.get("/api/v1/commodities/gas/prices", params=params)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list) and body
    first = body[0]
    assert first["instrument_id"] == "NG_HENRY_HUB"
    assert "price" in first


def test_lithium_endpoint_returns_supply_chain():
    today = date.today()
    params = {
        "material": "lithium_carbonate",
        "start_date": today.isoformat(),
        "end_date": today.isoformat(),
    }
    response = client.get("/api/v1/commodities/battery-materials/lithium", params=params)
    assert response.status_code == 200
    payload = response.json()
    assert payload["material"] == "lithium_carbonate"
    assert isinstance(payload["prices"], list) and payload["prices"]
    assert isinstance(payload["supply_chain"], list) and payload["supply_chain"]
