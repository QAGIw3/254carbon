"""
WTO Data Portal Connector

Coverage: Tariffs, trade policy, commerce.
Portal: https://timeseries.wto.org/

Production: Use WTO API where available; this scaffold emits a generic
tariff rate and trade policy index sample.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class WTODataConnector(Ingestor):
    """WTO data connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # WTO API (requires key): https://api.wto.org/timeseries/v1
        # Example: /v1?i=AV_TW_R&reporters=840&indicators=AV_TW_R&subscription-key=...
        self.portal_url = config.get("portal_url", "https://timeseries.wto.org/")
        self.country = config.get("country", "WLD")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "tariff_rate", "market": "economics", "product": "avg_tariff_pct", "unit": "%", "update_freq": "annual"},
                {"name": "trade_policy_index", "market": "economics", "product": "tpi_index", "unit": "index", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "avg_tariff_rate": "GET https://api.wto.org/timeseries/v1?i=AV_TW_R&reporters=840&subscription-key=...",
                "trade_policy_index": "GET https://api.wto.org/timeseries/v1?i=TRD_PRF&reporters=840&subscription-key=...",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.country, "variable": "avg_tariff_pct", "value": 7.4, "unit": "%"}
        yield {"timestamp": now, "entity": self.country, "variable": "tpi_index", "value": 65.2, "unit": "index"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WLD")
        variable = raw.get("variable", "avg_tariff_pct")
        instrument = f"WTO.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} WTO events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"WTO checkpoint saved: {state}")


if __name__ == "__main__":
    connector = WTODataConnector({"source_id": "wto_data"})
    connector.run()
