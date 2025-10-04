import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LOCAL_DEV", "true")

from platform.apps.gateway.main import app  # type: ignore


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_websocket_subscribe_dev(client: TestClient):
    with client.websocket_connect("/api/v1/stream") as ws:
        ws.send_json({
            "type": "subscribe",
            "instruments": ["MISO.HUB.INDIANA"],
            "commodities": ["oil"],
            "all": False,
            "api_key": "dev-key"
        })
        msg = ws.receive_json()
        assert msg["type"] == "subscribed"
        assert "instruments" in msg
        assert "commodities" in msg


def test_websocket_rejects_without_token_when_prod(client: TestClient, monkeypatch):
    # Simulate production mode
    monkeypatch.setenv("LOCAL_DEV", "false")
    with client.websocket_connect("/api/v1/stream") as ws:
        ws.send_json({
            "type": "subscribe",
            "instruments": ["MISO.HUB.INDIANA"],
            "api_key": ""
        })
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Unauthorized" in msg["message"] or "Invalid" in msg["message"]

