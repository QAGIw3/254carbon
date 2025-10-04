"""Synthetic AIS vessel and port call connector for LNG supply chain analytics."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from math import sin
from typing import Any, Dict, Iterator, Optional

from .base import (
    GeoLocation,
    InfrastructureConnector,
    InfrastructureType,
    LNGTerminal,
    OperationalStatus,
)
from ....base import CommodityType

logger = logging.getLogger(__name__)


class AISMaritimeConnector(InfrastructureConnector):
    """Connector that normalises AIS vessel movements around LNG terminals."""

    DEFAULT_PORTS: Dict[str, Dict[str, Any]] = {
        "US_SABINE_PASS": {"country": "US", "region": "TX", "lat": 29.738, "lon": -93.871},
        "US_CORPUS_CHRISTI": {"country": "US", "region": "TX", "lat": 27.826, "lon": -97.390},
        "QA_RAS_LAFFAN": {"country": "QA", "region": "QA", "lat": 25.914, "lon": 51.634},
        "BE_ZEEBRUGGE": {"country": "BE", "region": "BE", "lat": 51.332, "lon": 3.197},
        "NL_GATE": {"country": "NL", "region": "NL", "lat": 51.945, "lon": 4.119},
    }

    METRICS: Dict[str, str] = {
        "lng_vessels_at_berth_count": "count",
        "lng_vessels_waiting_count": "count",
        "lng_daily_port_calls": "count",
        "lng_average_loading_duration_hours": "hours",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.source_id:
            self.source_id = "ais_maritime"

        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")

        self.commodity_type = CommodityType.GAS
        self.lookback_hours = int(config.get("lookback_hours", 24))
        self.synthetic = bool(config.get("synthetic", True))
        self.port_catalog = config.get("ports") or self.DEFAULT_PORTS
        self._initialize_assets()
        self._state = self.load_checkpoint() or {}

    def _initialize_assets(self) -> None:
        """Initialise LNG terminal assets used for AIS derived metrics."""
        now = datetime.utcnow().date()
        for port_code, meta in self.port_catalog.items():
            location = GeoLocation(meta["lat"], meta["lon"])
            asset = LNGTerminal(
                asset_id=port_code,
                name=port_code.replace("_", " "),
                asset_type=InfrastructureType.LNG_TERMINAL,
                location=location,
                country=meta["country"],
                region=meta.get("region"),
                operator=meta.get("operator"),
                status=OperationalStatus.OPERATIONAL,
                storage_capacity_gwh=meta.get("storage_capacity_gwh", 600.0),
                regasification_capacity_gwh_d=meta.get("regas_capacity_gwh_d", 30.0),
                commissioned_date=meta.get("commissioned_date", now - timedelta(days=365 * 5)),
            )
            self.assets[port_code] = asset

    def discover(self) -> Dict[str, Any]:
        """Describe exposed AIS metrics."""
        return {
            "source_id": self.source_id,
            "coverage": list(self.port_catalog.keys()),
            "streams": [
                {
                    "name": "lng_port_activity",
                    "metrics": list(self.METRICS.keys()),
                    "frequency": "hourly",
                }
            ],
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Yield AIS-derived metrics for each configured LNG terminal."""
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        for port_code, asset in self.assets.items():
            phase = (hash(port_code) % 12) / 12
            hours_since_midnight = now.hour + phase
            for metric, unit in self.METRICS.items():
                value = self._synthesise_metric(metric, hours_since_midnight)
                metadata = {
                    "port_code": port_code,
                    "metric_origin": "ais_synthetic" if self.synthetic else "ais_feed",
                }
                yield {
                    "asset": asset,
                    "metric": metric,
                    "unit": unit,
                    "value": value,
                    "event_time": now,
                    "metadata": metadata,
                }
        self._state["last_event_time"] = now.isoformat()
        self.checkpoint(self._state)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        asset = raw["asset"]
        return self.create_infrastructure_event(
            asset=asset,
            metric=raw["metric"],
            value=float(raw["value"]),
            unit=raw["unit"],
            event_time=raw["event_time"],
            metadata=raw.get("metadata"),
        )

    def checkpoint(self, state: Dict[str, Any]) -> None:
        super().checkpoint(state)

    def _synthesise_metric(self, metric: str, hours_since_midnight: float) -> float:
        """Create deterministic but realistic-looking metric values."""
        baseline = {
            "lng_vessels_at_berth_count": 3,
            "lng_vessels_waiting_count": 2,
            "lng_daily_port_calls": 4,
            "lng_average_loading_duration_hours": 18,
        }.get(metric, 1)
        amplitude = {
            "lng_vessels_at_berth_count": 1.2,
            "lng_vessels_waiting_count": 1.0,
            "lng_daily_port_calls": 1.5,
            "lng_average_loading_duration_hours": 2.0,
        }.get(metric, 0.5)
        value = baseline + amplitude * (sin(hours_since_midnight) + 1)
        if "duration" in metric:
            return round(max(value, 1.0), 2)
        return round(max(value, 0.0))


__all__ = ["AISMaritimeConnector"]
