"""
FAO FAOSTAT Connector

Coverage: Global agriculture, food production, land use.
Portal: https://www.fao.org/faostat/en/

Production: Use FAOSTAT API for domain-specific datasets; this scaffold emits
sample production and land use metrics.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class FAOSTATConnector(Ingestor):
    """FAOSTAT agriculture/land use connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # FAOSTAT REST base: https://fenixservices.fao.org/faostat/api/v1/en
        # Examples: /FAOSTAT/Agri_Production?item_code=2511&area_code=2
        self.api_base = config.get("api_base", "https://fenixservices.fao.org/faostat/api/v1/en")
        self.country = config.get("country", "WLD")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "cereal_production", "market": "geospatial", "product": "cereal_prod_t", "unit": "t", "update_freq": "annual"},
                {"name": "agri_land_share", "market": "geospatial", "product": "agri_land_pct", "unit": "%", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "cereal_production": "GET {base}/FAOSTAT/Agri_Production?item_code=2511&area_code=2",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.country, "variable": "cereal_prod_t", "value": 2_825_000_000.0, "unit": "t"}
        yield {"timestamp": now, "entity": self.country, "variable": "agri_land_pct", "value": 37.8, "unit": "%"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WLD")
        variable = raw.get("variable", "cereal_prod_t")
        instrument = f"FAOSTAT.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} FAOSTAT events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"FAOSTAT checkpoint saved: {state}")


if __name__ == "__main__":
    connector = FAOSTATConnector({"source_id": "faostat"})
    connector.run()
