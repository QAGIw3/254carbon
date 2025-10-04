"""
IMF Data Connector

Coverage: Macroeconomic, financial, trade statistics.
Portal: https://data.imf.org/

Production: Use IMF SDMX API as available; this scaffold emits generic macro
indicators.

Data Flow
---------
IMF SDMX API/CSV → indicator series → canonical economics → Kafka
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class IMFConnector(Ingestor):
    """IMF macroeconomic data connector (limited API)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # IMF SDMX JSON: https://dataservices.imf.org/REST/SDMX_JSON.svc
        # Example: /CompactData/IFS/B..PCPI_IX?startPeriod=2020&endPeriod=2024
        self.api_base = config.get("api_base", "https://dataservices.imf.org/REST/SDMX_JSON.svc")
        self.country = config.get("country", "WLD")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "current_account", "market": "economics", "product": "current_account_pct_gdp", "unit": "%", "update_freq": "annual"},
                {"name": "gov_debt", "market": "economics", "product": "gov_debt_pct_gdp", "unit": "%", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "inflation_index": "GET {base}/CompactData/IFS/{country}.PCPI_IX?startPeriod=2020&endPeriod=2024",
                "gov_debt": "GET {base}/CompactData/GFS/{country}.GGXWDG_NGDP?startPeriod=2020&endPeriod=2024",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.country, "variable": "current_account_pct_gdp", "value": -1.8, "unit": "%"}
        yield {"timestamp": now, "entity": self.country, "variable": "gov_debt_pct_gdp", "value": 97.0, "unit": "%"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WLD")
        variable = raw.get("variable", "current_account_pct_gdp")
        instrument = f"IMF.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} IMF events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"IMF checkpoint saved: {state}")


if __name__ == "__main__":
    connector = IMFConnector({"source_id": "imf_data"})
    connector.run()
