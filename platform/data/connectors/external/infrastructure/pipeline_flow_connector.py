"""Natural gas pipeline flow connector normalising capacity and utilisation signals."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator

from .base import (
    GeoLocation,
    InfrastructureAsset,
    InfrastructureConnector,
    InfrastructureType,
    OperationalStatus,
)
from ....base import CommodityType

logger = logging.getLogger(__name__)


class PipelineFlowConnector(InfrastructureConnector):
    """Connector for pipeline flow and capacity fundamentals."""

    DEFAULT_PIPELINES: Dict[str, Dict[str, Any]] = {
        "TCO_MAINLINE": {
            "name": "Columbia Gas Transmission Mainline",
            "country": "US",
            "region": "US_NE",
            "lat": 39.4,
            "lon": -80.0,
            "capacity_bcfd": 7.8,
        },
        "TGP_ZONE4": {
            "name": "Tennessee Gas Pipeline Zone 4",
            "country": "US",
            "region": "US_NE",
            "lat": 41.0,
            "lon": -74.0,
            "capacity_bcfd": 5.2,
        },
        "NGPL_GULFCOAST": {
            "name": "NGPL Gulf Coast",
            "country": "US",
            "region": "US_GC",
            "lat": 29.6,
            "lon": -95.2,
            "capacity_bcfd": 4.4,
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.source_id:
            self.source_id = "pipeline_flows"

        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")

        self.commodity_type = CommodityType.GAS
        self.pipeline_catalog = config.get("pipelines") or self.DEFAULT_PIPELINES
        self.lookback_hours = int(config.get("lookback_hours", 24))
        self._state = self.load_checkpoint() or {}
        self._initialize_assets()

    def _initialize_assets(self) -> None:
        today = datetime.utcnow().date()
        for pipeline_id, meta in self.pipeline_catalog.items():
            asset = InfrastructureAsset(
                asset_id=pipeline_id,
                name=meta.get("name", pipeline_id.replace("_", " ")),
                asset_type=InfrastructureType.INTERCONNECTOR,
                location=GeoLocation(meta["lat"], meta["lon"]),
                country=meta.get("country", "US"),
                region=meta.get("region"),
                status=OperationalStatus.OPERATIONAL,
                commissioned_date=today - timedelta(days=365 * 10),
                metadata={"capacity_bcfd": meta.get("capacity_bcfd")},
            )
            self.assets[pipeline_id] = asset

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "pipelines": list(self.pipeline_catalog.keys()),
            "metrics": ["pipeline_flow_bcfd", "pipeline_capacity_bcfd"],
            "frequency": "hourly",
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        for pipeline_id, asset in self.assets.items():
            meta = self.pipeline_catalog[pipeline_id]
            capacity = float(meta.get("capacity_bcfd", 5.0))
            flow = self._synthetic_flow(pipeline_id, now)
            utilisation = min(flow / capacity, 1.2)

            yield {
                "asset": asset,
                "metric": "pipeline_flow_bcfd",
                "value": flow,
                "unit": "Bcf/d",
                "event_time": now,
                "metadata": {"pipeline_id": pipeline_id, "utilisation_pct": utilisation * 100},
            }
            yield {
                "asset": asset,
                "metric": "pipeline_capacity_bcfd",
                "value": capacity,
                "unit": "Bcf/d",
                "event_time": now,
                "metadata": {"pipeline_id": pipeline_id},
            }
        self._state["last_event_time"] = now.isoformat()
        self.checkpoint(self._state)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return self.create_infrastructure_event(
            asset=raw["asset"],
            metric=raw["metric"],
            value=float(raw["value"]),
            unit=raw["unit"],
            event_time=raw["event_time"],
            metadata=raw.get("metadata"),
        )

    def checkpoint(self, state: Dict[str, Any]) -> None:
        super().checkpoint(state)

    def _synthetic_flow(self, pipeline_id: str, as_of: datetime) -> float:
        base = float(self.pipeline_catalog[pipeline_id].get("capacity_bcfd", 5.0)) * 0.78
        daily_wave = 0.12 * base * ((as_of.hour % 24) / 24)
        seasonal_wave = 0.08 * base * ((as_of.timetuple().tm_yday % 90) / 90)
        return round(base + daily_wave + seasonal_wave, 3)


__all__ = ["PipelineFlowConnector"]
