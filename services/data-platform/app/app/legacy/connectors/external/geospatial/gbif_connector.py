"""
GBIF Connector

Coverage: Global Biodiversity Information Facility — biodiversity, species distribution.
Portal: https://www.gbif.org/

Production: Use GBIF API for species/occurrence queries; this scaffold emits
species occurrence counts and biodiversity indices.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class GBIFConnector(Ingestor):
    """GBIF biodiversity connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # GBIF REST base: https://api.gbif.org/v1
        # Example: /occurrence/search?country=US&taxon_key=212&limit=300
        self.api_base = config.get("api_base", "https://api.gbif.org/v1")
        self.region = config.get("region", "WORLD")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "occurrence_count", "market": "geospatial", "product": "occurrences", "unit": "count", "update_freq": "ad-hoc"},
                {"name": "biodiversity_index", "market": "geospatial", "product": "biodiv_index", "unit": "index", "update_freq": "ad-hoc"},
            ],
            "endpoint_examples": {
                "occurrence_search": "GET {base}/occurrence/search?country=US&limit=1",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.region, "variable": "occurrences", "value": 125_000_000.0, "unit": "count"}
        yield {"timestamp": now, "entity": self.region, "variable": "biodiv_index", "value": 0.73, "unit": "index"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WORLD")
        variable = raw.get("variable", "occurrences")
        instrument = f"GBIF.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} GBIF events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"GBIF checkpoint saved: {state}")


if __name__ == "__main__":
    connector = GBIFConnector({"source_id": "gbif"})
    connector.run()
