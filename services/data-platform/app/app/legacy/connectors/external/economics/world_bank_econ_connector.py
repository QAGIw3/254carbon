"""
World Bank Open Data (Economics) Connector

Coverage: Global development and economic indicators.
Portal: https://data.worldbank.org/

Production: Integrate with World Bank API using indicator codes.
This scaffold emits GDP per capita and inflation for a selected country.

Data Flow
---------
World Bank API → indicator series → canonical economics → Kafka
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from kafka import KafkaProducer
import json
import requests

from ...base import Ingestor

logger = logging.getLogger(__name__)


class WorldBankEconomicsConnector(Ingestor):
    """World Bank economics indicators connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # World Bank API base: https://api.worldbank.org/v2
        # Example: /country/US/indicator/FP.CPI.TOTL.ZG?date=2020:2024&format=json
        self.api_base = config.get("api_base", "https://api.worldbank.org/v2")
        self.country = config.get("country", "WLD")
        self.live: bool = config.get("live", False)
        self.indicators: List[str] = config.get("indicators", ["NY.GDP.PCAP.CD", "FP.CPI.TOTL.ZG"])  # GDP pc USD, CPI inflation
        self.start_year: Optional[int] = config.get("start_year")
        self.end_year: Optional[int] = config.get("end_year")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "gdp_per_capita", "market": "economics", "product": "gdp_pc_usd", "unit": "USD", "update_freq": "annual"},
                {"name": "inflation", "market": "economics", "product": "inflation_pct", "unit": "%", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "gdp_pc": "GET {base}/country/{country}/indicator/NY.GDP.PCAP.CD?format=json",
                "inflation": "GET {base}/country/{country}/indicator/FP.CPI.TOTL.ZG?format=json",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self.country, "variable": "gdp_pc_usd", "value": 12_100.0, "unit": "USD"}
            yield {"timestamp": now, "entity": self.country, "variable": "inflation_pct", "value": 4.1, "unit": "%"}
            return

        start = self.start_year or (datetime.utcnow().year - 10)
        end = self.end_year or datetime.utcnow().year
        date_range = f"{start}:{end}"

        for ind in self.indicators:
            url = f"{self.api_base}/country/{self.country}/indicator/{ind}?date={date_range}&format=json&per_page=20000"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                series = data[1] if isinstance(data, list) and len(data) > 1 else []
            except Exception as e:
                logging.error(f"World Bank econ fetch failed: {e} ({ind})")
                continue

            for row in series:
                year = row.get("date")
                value = row.get("value")
                if year is None or value is None:
                    continue
                ts_iso = f"{year}-01-01T00:00:00Z"
                yield {
                    "timestamp": ts_iso,
                    "entity": self.country,
                    "variable": ind.lower(),
                    "value": float(value),
                    "unit": row.get("unit", ""),
                }

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WLD")
        variable = raw.get("variable", "gdp_pc_usd")
        instrument = f"WB.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} World Bank Economics events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"World Bank Economics checkpoint saved: {state}")


if __name__ == "__main__":
    connector = WorldBankEconomicsConnector({"source_id": "world_bank_econ"})
    connector.run()
