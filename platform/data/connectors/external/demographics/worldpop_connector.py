"""
WorldPop Connector

Coverage: High-resolution gridded global population maps.
Portal: https://www.worldpop.org/

Production: Typically uses download services (GeoTIFF) and raster stats.
This scaffold emits country-level summaries.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class WorldPopConnector(Ingestor):
    """WorldPop gridded population connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.portal_url = config.get("portal_url", "https://www.worldpop.org/")
        self.country = config.get("country", "WLD")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "population_density", "market": "demographics", "product": "pop_density", "unit": "people/km2", "update_freq": "annual"},
                {"name": "total_population", "market": "demographics", "product": "population", "unit": "people", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "wcs_getcoverage": (
                    "GET https://hub.worldpop.org/geoserver/POP/wcs?service=WCS&request=GetCoverage&version=2.0.1"
                    "&coverageId=ppp_2020_1km_Aggregated&subset=Lat(34,35)&subset=Long(-119,-118)&format=image/tiff"
                ),
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.country, "variable": "pop_density", "value": 58.1, "unit": "people/km2"}
        yield {"timestamp": now, "entity": self.country, "variable": "population", "value": 7_950_000_000.0, "unit": "people"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WLD")
        variable = raw.get("variable", "pop_density")
        instrument = f"WORLDPOP.{entity}.{variable.upper()}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "demographics",
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
        logger.info(f"Emitted {count} WorldPop events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"WorldPop checkpoint saved: {state}")


if __name__ == "__main__":
    connector = WorldPopConnector({"source_id": "worldpop"})
    connector.run()
