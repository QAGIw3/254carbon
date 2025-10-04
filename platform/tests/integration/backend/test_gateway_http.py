import os
import pytest
from fastapi.testclient import TestClient

# Ensure local dev mode for permissive behavior
os.environ.setdefault("LOCAL_DEV", "true")

from platform.apps.gateway.main import app  # type: ignore
from fastapi.testclient import TestClient as _TC


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_instruments_requires_auth(client: TestClient):
    resp = client.get("/api/v1/instruments")
    # Should be 403 or 401 depending on security setup
    assert resp.status_code in (401, 403)


def test_prices_ticks_requires_auth(client: TestClient):
    resp = client.get("/api/v1/prices/ticks", params={
        "instrument_id": ["MISO.HUB.INDIANA"],
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-02T00:00:00Z",
        "price_type": "mid"
    })
    assert resp.status_code in (401, 403)


def test_commodity_prices_requires_auth(client: TestClient):
    resp = client.get(
        "/api/v1/commodities/WTI/prices",
        params={
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "price_type": "mid",
            "limit": 10,
        },
    )
    assert resp.status_code in (401, 403)


def test_export_preview_requires_auth(client: TestClient):
    resp = client.get(
        "/api/v1/research/export/preview",
        params={
            "dataset_id": "energy_prices",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "limit": 5,
        },
    )
    assert resp.status_code in (401, 403)


def test_sse_requires_auth(client: TestClient):
    with _TC(app, headers={}) as c:
        resp = c.get("/api/v1/stream/sse", params={"instruments": ["MISO.HUB.INDIANA"]}, stream=True)
        # Unauthorized since no Authorization header
        assert resp.status_code in (401, 403)


