"""Spire AIS adapter implementing the AISProvider interface."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List

import requests

from .ais_provider_base import AISProvider, PortCallEvent, VesselPosition

logger = logging.getLogger(__name__)


class SpireAISAdapter(AISProvider):
    """Adapter for Spire AIS API.

    Requires `api_key` and optional `api_base` (default spire endpoint).
    """

    DEFAULT_API_BASE = "https://ais.spire.com/vessels"

    def __init__(self, api_key: str, api_base: str | None = None, timeout: float = 30.0):
        self.api_key = api_key
        self.api_base = (api_base or self.DEFAULT_API_BASE).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_positions(self, area: Dict[str, float]) -> Iterable[VesselPosition]:
        params = {
            "minLat": area["min_lat"],
            "maxLat": area["max_lat"],
            "minLon": area["min_lon"],
            "maxLon": area["max_lon"],
        }
        payload = self._get("positions", params)
        for item in payload.get("data", []):
            attrs = item.get("attributes", {})
            position = attrs.get("position", {})
            yield VesselPosition(
                imo=attrs.get("imo"),
                mmsi=str(attrs.get("mmsi")),
                latitude=position.get("lat"),
                longitude=position.get("lon"),
                speed_knots=position.get("speed", 0.0),
                course_deg=position.get("course"),
                heading_deg=position.get("heading"),
                timestamp=self._parse_time(position.get("timestamp")),
                status=attrs.get("navigationalStatus"),
                destination=attrs.get("destination"),
            )

    def fetch_port_calls(self, terminal_ids: List[str], lookback_hours: int) -> Iterable[PortCallEvent]:
        since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
        params = {"terminalIds": ",".join(terminal_ids), "since": since}
        payload = self._get("port-calls", params)
        for item in payload.get("data", []):
            attrs = item.get("attributes", {})
            yield PortCallEvent(
                imo=attrs.get("imo"),
                mmsi=str(attrs.get("mmsi")),
                terminal_id=attrs.get("terminalId"),
                event_type=attrs.get("eventType"),
                event_time=self._parse_time(attrs.get("eventTime")),
                status=attrs.get("status"),
                metadata={
                    "source": attrs.get("source"),
                    "port": attrs.get("port"),
                },
            )

    def _get(self, endpoint: str, params: Dict[str, object]) -> Dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_base}/{endpoint}"
        resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _parse_time(self, value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
