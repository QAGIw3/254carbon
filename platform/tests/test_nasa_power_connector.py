import json
from typing import Dict

import types

from platform.data.connectors.external.weather.nasa_power_connector import NASAPowerConnector


class DummyResp:
    def __init__(self, payload: Dict):
        self._payload = payload

    def raise_for_status(self):
        return

    def json(self):
        return self._payload


def test_nasa_power_mock_mode_maps_events(monkeypatch):
    cfg = {"source_id": "nasa_power_test", "live": False}
    c = NASAPowerConnector(cfg)
    events = list(c.pull_or_subscribe())
    assert any(e["variable"] == "temp_c" for e in events)
    mapped = c.map_to_schema(events[0])
    assert mapped["market"] == "weather"
    assert "instrument_id" in mapped
    assert "value" in mapped


def test_nasa_power_live_mode_parses_series(monkeypatch):
    payload = {
        "properties": {
            "parameter": {
                "T2M": {"2024010100": 10.0},
                "WS10M": {"2024010100": 2.5},
                "ALLSKY_SFC_SW_DWN": {"2024010100": 1.2},
            }
        }
    }

    import platform.data.connectors.external.weather.nasa_power_connector as mod

    def fake_get(url, timeout=30):
        return DummyResp(payload)

    monkeypatch.setattr(mod.requests, "get", fake_get)

    cfg = {
        "source_id": "nasa_power_test",
        "live": True,
        "latitude": 34.05,
        "longitude": -118.24,
        "start": "20240101",
        "end": "20240101",
    }
    c = NASAPowerConnector(cfg)
    events = list(c.pull_or_subscribe())
    # Expect 3 variables * 1 timestamp
    assert len(events) == 3
    vars_seen = {e["variable"] for e in events}
    assert {"temp_c", "wind_ms", "ghi_mj_m2_hr"}.issubset(vars_seen)

