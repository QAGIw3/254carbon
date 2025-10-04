"""
UN Comtrade Connector

Coverage: International import/export flows.
Portal: https://comtradeplus.un.org/

Production: Use the Comtrade API for trade flows by HS codes; this scaffold
emits a generic trade balance and export/import totals.

Data Flow
---------
Comtrade API/CSV → trade flows → canonical economics → Kafka
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class UNComtradeConnector(Ingestor):
    """UN Comtrade trade flows connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # UN Comtrade API (Plus): https://comtradeapi.un.org/
        # Example: /public/v1/preview/flow?type=C&freq=M&px=HS&ps=2024&r=840&p=124&rg=all&cc=TOTAL
        self.api_base = config.get("api_base", "https://comtradeapi.un.org/public/v1")
        self.country = config.get("country", "WLD")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "exports", "market": "economics", "product": "exports_usd_bn", "unit": "USD bn", "update_freq": "monthly"},
                {"name": "imports", "market": "economics", "product": "imports_usd_bn", "unit": "USD bn", "update_freq": "monthly"},
                {"name": "trade_balance", "market": "economics", "product": "trade_balance_usd_bn", "unit": "USD bn", "update_freq": "monthly"},
            ],
            "endpoint_examples": {
                "preview_flow": (
                    "GET {base}/preview/flow?type=C&freq=M&px=HS&ps=2024&r=840&p=124&rg=all&cc=TOTAL"
                ),
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        exports = 2100.0
        imports = 2200.0
        yield {"timestamp": now, "entity": self.country, "variable": "exports_usd_bn", "value": exports, "unit": "USD bn"}
        yield {"timestamp": now, "entity": self.country, "variable": "imports_usd_bn", "value": imports, "unit": "USD bn"}
        yield {"timestamp": now, "entity": self.country, "variable": "trade_balance_usd_bn", "value": exports - imports, "unit": "USD bn"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WLD")
        variable = raw.get("variable", "trade_balance_usd_bn")
        instrument = f"COMTRADE.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} UN Comtrade events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"UN Comtrade checkpoint saved: {state}")


if __name__ == "__main__":
    connector = UNComtradeConnector({"source_id": "un_comtrade"})
    connector.run()
