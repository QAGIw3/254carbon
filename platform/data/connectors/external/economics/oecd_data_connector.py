"""
OECD Data Connector

Coverage: Labor, trade, environment, macro indicators.
Portal: https://data.oecd.org/

Production: Use OECD SDMX/JSON APIs; this scaffold emits unemployment and
trade balance metrics for integration testing.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class OECDDataConnector(Ingestor):
    """OECD broad data connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # OECD SDMX-JSON base: https://stats.oecd.org/sdmx-json
        # Example: /data/MEI_CLI/USA.ML.M?contentType=csv
        self.api_base = config.get("api_base", "https://stats.oecd.org/sdmx-json")
        self.country = config.get("country", "OECD")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "unemployment_rate", "market": "economics", "product": "unemp_rate", "unit": "%", "update_freq": "monthly"},
                {"name": "trade_balance", "market": "economics", "product": "trade_balance_usd_bn", "unit": "USD bn", "update_freq": "monthly"},
            ],
            "endpoint_examples": {
                "cli": "GET {base}/data/MEI_CLI/USA.ML.M?contentType=csv",
                "unemployment": "GET {base}/data/UNEMPLOY/USA.M.A?contentType=csv",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.country, "variable": "unemp_rate", "value": 5.8, "unit": "%"}
        yield {"timestamp": now, "entity": self.country, "variable": "trade_balance_usd_bn", "value": -12.3, "unit": "USD bn"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "OECD")
        variable = raw.get("variable", "unemp_rate")
        instrument = f"OECD.{entity}.{variable.upper()}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "economics",
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
        logger.info(f"Emitted {count} OECD Data events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"OECD Data checkpoint saved: {state}")


if __name__ == "__main__":
    connector = OECDDataConnector({"source_id": "oecd_data"})
    connector.run()
