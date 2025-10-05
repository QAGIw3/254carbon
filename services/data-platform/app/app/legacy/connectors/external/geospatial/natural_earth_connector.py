"""
Natural Earth Data Connector

Coverage: Boundaries, population centers, rivers, global maps.
Portal: https://www.naturalearthdata.com/

Production: Download shapefiles/GeoJSON; this scaffold emits metadata counts
for integration testing.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class NaturalEarthConnector(Ingestor):
    """Natural Earth global boundaries connector (download-based)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.portal_url = config.get("portal_url", "https://www.naturalearthdata.com/")
        self.scale = config.get("scale", "110m")  # 110m/50m/10m
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "admin_0_countries", "market": "geospatial", "product": "country_count", "unit": "count", "update_freq": "ad-hoc"},
                {"name": "populated_places", "market": "geospatial", "product": "pop_places", "unit": "count", "update_freq": "ad-hoc"},
            ],
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.scale, "variable": "country_count", "value": 215.0, "unit": "count"}
        yield {"timestamp": now, "entity": self.scale, "variable": "pop_places", "value": 7267.0, "unit": "count"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "110m")
        variable = raw.get("variable", "country_count")
        instrument = f"NE.{entity}.{variable.upper()}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "geospatial",
            "product": variable,
            "instrument_id": instrument,
            "location_code": instrument,
            "price_type": "observation",
            "value": float(raw.get("value", 0.0)),
            "volume": None,
            "currency": "USD",
            "unit": raw.get("unit", "unit"),
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }

    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        if self.producer is None:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        count = 0
        for event in events:
            try:
                self.producer.send(self.kafka_topic, value=event)
                count += 1
            except Exception as e:
                logger.error(f"Kafka send error: {e}")
        if self.producer is not None:
            self.producer.flush()
        logger.info(f"Emitted {count} Natural Earth events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"Natural Earth checkpoint saved: {state}")


if __name__ == "__main__":
    connector = NaturalEarthConnector({"source_id": "natural_earth"})
    connector.run()

