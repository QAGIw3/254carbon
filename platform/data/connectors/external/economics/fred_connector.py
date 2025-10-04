"""
FRED (Federal Reserve Bank of St. Louis) Connector

Coverage: US + global macro indicators: GDP, CPI, interest rates.
Portal: https://fred.stlouisfed.org/

Production: Integrate via FRED API (https://api.stlouisfed.org/fred/).
This scaffold emits representative GDP and CPI values.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class FREDConnector(Ingestor):
    """FRED macroeconomic indicators connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # FRED API base: https://api.stlouisfed.org/fred
        # Example: /series/observations?series_id=CPIAUCSL&api_key=...&file_type=json
        self.api_base = config.get("api_base", "https://api.stlouisfed.org/fred")
        self.api_key = config.get("api_key")
        self.country = config.get("country", "US")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "gdp", "market": "economics", "product": "gdp_usd_bn", "unit": "USD bn", "update_freq": "quarterly"},
                {"name": "cpi", "market": "economics", "product": "cpi_index", "unit": "index", "update_freq": "monthly"},
                {"name": "interest_rate", "market": "economics", "product": "fed_funds_rate", "unit": "%", "update_freq": "daily"},
            ],
            "endpoint_examples": {
                "cpi": "GET {base}/series/observations?series_id=CPIAUCSL&api_key=...&file_type=json",
                "gdp": "GET {base}/series/observations?series_id=GDP&api_key=...&file_type=json",
                "fed_funds": "GET {base}/series/observations?series_id=FEDFUNDS&api_key=...&file_type=json",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.country, "variable": "gdp_usd_bn", "value": 26_000.0, "unit": "USD bn"}
        yield {"timestamp": now, "entity": self.country, "variable": "cpi_index", "value": 304.1, "unit": "index"}
        yield {"timestamp": now, "entity": self.country, "variable": "fed_funds_rate", "value": 5.5, "unit": "%"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "US")
        variable = raw.get("variable", "gdp_usd_bn")
        instrument = f"FRED.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} FRED events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"FRED checkpoint saved: {state}")


if __name__ == "__main__":
    connector = FREDConnector({"source_id": "fred"})
    connector.run()
