"""
LNG Charter Rate Connector

Overview
--------
Publishes benchmark LNG vessel charter rates (e.g., TFDE, MEGI, FSRU) into the
fundamentals topic for downstream routing and analytics. This scaffold uses a
small deterministic time series for development; plug in a broker or index
provider in production.

Data Flow
---------
Provider (indices) → normalize per series → canonical fundamentals → Kafka

Configuration
-------------
- `series`: Mapping of series code to metadata (unit, route, vessel_class).
- `lookback_days`: Backfill window emitted each run.
- `kafka.topic`/`kafka.bootstrap_servers`: Emission settings.

Operational Notes
-----------------
- Connector emits one value per series per day across the lookback window,
  then checkpoints the last date for monitoring.
- Customize or extend `DEFAULT_SERIES` as needed to match provider coverage.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, Optional

from ....base import CommodityType, Ingestor

logger = logging.getLogger(__name__)


class CharterRateConnector(Ingestor):
    """Publish LNG charter rate benchmarks into the fundamentals topic.

    Series values represent assessed daily rate levels. Units default to
    `USD/DAY` but can be overridden per series.
    """

    DEFAULT_SERIES: Dict[str, Dict[str, Any]] = {
        "TFDE_SPOT": {
            "description": "Tri-fuel diesel electric LNG carrier spot rate",
            "unit": "USD/DAY",
            "route": "Atlantic Basin",
            "vessel_class": "TFDE",
        },
        "MEGI_SPOT": {
            "description": "MEGI LNG carrier spot rate",
            "unit": "USD/DAY",
            "route": "Pacific Basin",
            "vessel_class": "MEGI",
        },
        "FSRU_CHARTER": {
            "description": "Floating storage regasification unit lease rate",
            "unit": "USD/DAY",
            "route": "Global",
            "vessel_class": "FSRU",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.source_id:
            self.source_id = "lng_charter_rates"

        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")

        self.series: Dict[str, Dict[str, Any]] = config.get("series") or self.DEFAULT_SERIES
        self.lookback_days = int(config.get("lookback_days", 7))
        self.commodity_type = CommodityType.GAS
        self._state = self.load_checkpoint() or {}

    def discover(self) -> Dict[str, Any]:
        """Describe available charter rate series for inspection/telemetry."""
        return {
            "source_id": self.source_id,
            "series": [
                {"code": code, **spec}
                for code, spec in self.series.items()
            ],
            "frequency": "daily",
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Yield daily charter rate benchmarks over the lookback window."""
        anchor = datetime.utcnow().date()
        for offset in range(self.lookback_days):
            day = anchor - timedelta(days=offset)
            for code, spec in self.series.items():
                value = self._synthetic_value(code, day.toordinal())
                yield {
                    "date": day,
                    "code": code,
                    "spec": spec,
                    "value": value,
                }
        self._state["last_event_date"] = anchor.isoformat()
        self.checkpoint(self._state)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map synthetic/raw series to the canonical event wire format."""
        event_time = datetime.combine(raw["date"], datetime.min.time(), tzinfo=timezone.utc)
        spec = raw["spec"]
        instrument_id = f"CHARTER.{raw['code']}"
        metadata = {
            "description": spec.get("description"),
            "route": spec.get("route"),
            "vessel_class": spec.get("vessel_class"),
        }
        return {
            "event_time_utc": int(event_time.timestamp() * 1000),
            "market": "fundamentals",
            "product": "lng_charter_rate",
            "instrument_id": instrument_id,
            "location_code": spec.get("route", "GLOBAL"),
            "price_type": "observation",
            "value": float(raw["value"]),
            "unit": spec.get("unit", "USD/DAY"),
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "metadata": metadata,
        }

    def checkpoint(self, state: Dict[str, Any]) -> None:
        super().checkpoint(state)

    def _synthetic_value(self, code: str, ordinal: int) -> float:
        base = {
            "TFDE_SPOT": 120_000,
            "MEGI_SPOT": 135_000,
            "FSRU_CHARTER": 160_000,
        }.get(code, 100_000)
        seasonal = 1.0 + 0.05 * ((ordinal % 30) / 30)
        return round(base * seasonal, 2)


__all__ = ["CharterRateConnector"]
