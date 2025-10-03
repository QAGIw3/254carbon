import os
import pytest
from fastapi.testclient import TestClient
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MAIN_PATH = ROOT / "platform" / "apps" / "report-service" / "main.py"

spec = importlib.util.spec_from_file_location("report_service_main", str(MAIN_PATH))
module = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore
app = module.app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


