"""
World Bank Energy Data Connector

Coverage: Global energy indicators and infrastructure.
Catalog: https://datacatalog.worldbank.org/

This scaffold emits sample indicators; production will use the World Bank
API (e.g., https://api.worldbank.org/v2/) with indicator codes.

Data Flow
---------
World Bank API → indicator series → canonical fundamentals → Kafka
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

from kafka import KafkaProducer
import json
import requests

from ...base import Ingestor

logger = logging.getLogger(__name__)


class WorldBankEnergyConnector(Ingestor):
    """World Bank energy indicators connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # World Bank API base: https://api.worldbank.org/v2
        # Example: /country/WLD/indicator/EG.USE.PCAP.KG.OE?date=2000:2024&format=json
        self.api_base = config.get("api_base", "https://api.worldbank.org/v2")
        self.indicators: List[str] = config.get("indicators", ["EG.USE.PCAP.KG.OE", "EG.ELC.ACCS.ZS"])  # per-capita use, access
        self.country = config.get("country", "WLD")  # World aggregate
        self.live: bool = config.get("live", False)
        self.start_year: Optional[int] = config.get("start_year")
        self.end_year: Optional[int] = config.get("end_year")
        # Optional aliases/units override for indicators
        # { 'EG.USE.PCAP.KG.OE': ['energy_use_pc','kg_oil_eq'] }
        self.indicator_aliases: Dict[str, Tuple[str, str]] = config.get("indicator_aliases", {
            "EG.USE.PCAP.KG.OE": ("energy_use_pc", "kg_oil_eq"),
            "EG.ELC.ACCS.ZS": ("elec_access_pct", "%"),
        })
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "energy_use_per_capita", "market": "infra", "product": "energy_use_pc", "unit": "kg_oil_eq", "update_freq": "annual"},
                {"name": "electricity_access", "market": "infra", "product": "elec_access_pct", "unit": "%", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "energy_use_pc": "GET {base}/country/{country}/indicator/EG.USE.PCAP.KG.OE?format=json",
                "elec_access": "GET {base}/country/{country}/indicator/EG.ELC.ACCS.ZS?format=json",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self.country, "variable": "energy_use_pc", "value": 1800.0, "unit": "kg_oil_eq"}
            yield {"timestamp": now, "entity": self.country, "variable": "elec_access_pct", "value": 88.5, "unit": "%"}
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
                logging.error(f"World Bank indicator fetch failed: {e} ({ind})")
                continue

            for row in series:
                year = row.get("date")
                value = row.get("value")
                if year is None or value is None:
                    continue
                ts_iso = f"{year}-01-01T00:00:00Z"
                var, unit = self.indicator_aliases.get(ind, (ind.lower(), row.get("unit", "")))
                yield {
                    "timestamp": ts_iso,
                    "entity": self.country,
                    "variable": var,
                    "value": float(value),
                    "unit": unit,
                }

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WLD")
        variable = raw.get("variable", "energy_use_pc")
        instrument = f"WB.{entity}.{variable.upper()}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "infra",
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
        logger.info(f"Emitted {count} World Bank Energy events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"World Bank Energy checkpoint saved: {state}")


if __name__ == "__main__":
    connector = WorldBankEnergyConnector({"source_id": "world_bank_energy"})
    connector.run()
