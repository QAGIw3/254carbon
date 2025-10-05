"""
Rail and Trucking Tariff Connector (Coal Logistics)

Overview
--------
Publishes regional rail and trucking tariff indices used in coal logistics
models. Values are representative and deterministic for development; wire to a
transportation index provider in production.

Data Flow
---------
Provider (indices) → normalize per region/mode → canonical fundamentals → Kafka

Configuration
-------------
- `series`: Dict keyed by series_id with `mode`, `region`, `unit`, `baseline`.
- `lookback_days`: Number of days to emit per run for backfill.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- Emissions occur at daily granularity; downstream can resample as needed.
- The synthetic generator adds small seasonal drift and weekend adjustment.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator

from ....base import CommodityType, Ingestor

logger = logging.getLogger(__name__)


class TransportTariffConnector(Ingestor):
    """Publishes regional rail and trucking tariff indices.

    Output products are named `<mode>_tariff_index` and instrument IDs encode
    the mode and region for easy filtering in analytics.
    """

    DEFAULT_SERIES: Dict[str, Dict[str, Any]] = {
        "US_RAIL_APPALACHIA": {
            "mode": "rail",
            "region": "US_APP",
            "unit": "USD/TONNE_MILE",
            "baseline": 0.032,
        },
        "US_TRUCK_GULF": {
            "mode": "truck",
            "region": "US_GC",
            "unit": "USD/TONNE_MILE",
            "baseline": 0.082,
        },
        "CN_RAIL_NORTH": {
            "mode": "rail",
            "region": "CN_NORTH",
            "unit": "USD/TONNE_MILE",
            "baseline": 0.048,
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.source_id:
            self.source_id = "transport_tariffs"

        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")

        self.series = config.get("series") or self.DEFAULT_SERIES
        self.lookback_days = int(config.get("lookback_days", 14))
        self.commodity_type = CommodityType.COAL
        self._state = self.load_checkpoint() or {}

    def discover(self) -> Dict[str, Any]:
        """Describe series and modes for telemetry and UI inspection."""
        return {
            "source_id": self.source_id,
            "series": list(self.series.keys()),
            "frequency": "weekly",
            "modes": list({spec["mode"] for spec in self.series.values()}),
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Yield a daily value for each series across the lookback window."""
        anchor = datetime.utcnow().date()
        for offset in range(self.lookback_days):
            date = anchor - timedelta(days=offset)
            for series_id, spec in self.series.items():
                yield {
                    "series_id": series_id,
                    "spec": spec,
                    "date": date,
                    "value": self._synthetic_value(series_id, date.toordinal(), spec["baseline"]),
                }
        self._state["last_event_date"] = anchor.isoformat()
        self.checkpoint(self._state)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map synthetic/raw tariff values to the canonical event format."""
        event_time = datetime.combine(raw["date"], datetime.min.time(), tzinfo=timezone.utc)
        spec = raw["spec"]
        product = f"{spec['mode']}_tariff_index"
        instrument_id = f"{spec['mode'].upper()}_{spec['region']}"
        metadata = {
            "region": spec["region"],
            "mode": spec["mode"],
        }
        return {
            "event_time_utc": int(event_time.timestamp() * 1000),
            "market": "fundamentals",
            "product": product,
            "instrument_id": instrument_id,
            "location_code": spec["region"],
            "price_type": "index",
            "value": float(raw["value"]),
            "unit": spec["unit"],
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "metadata": metadata,
        }

    def checkpoint(self, state: Dict[str, Any]) -> None:
        super().checkpoint(state)

    def _synthetic_value(self, series_id: str, ordinal: int, baseline: float) -> float:
        """Create a simple drift + weekend-adjusted tariff index."""
        drift = 1 + 0.01 * ((ordinal % 365) / 365)
        weekend_adjustment = 0.98 if ordinal % 7 in (5, 6) else 1.0
        return round(baseline * drift * weekend_adjustment, 5)


__all__ = ["TransportTariffConnector"]
