"""
UN Data Connector

Coverage: Global population, health, education, economy.
Portal: https://data.un.org/

Live mode supports download-driven ingestion (CSV) and an optional SDMX path
via UN SDMX endpoints (where available). Safe mocks are emitted if live=false.
"""
import csv
import io
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from kafka import KafkaProducer
import json
import requests

from ...base import Ingestor

logger = logging.getLogger(__name__)


class UNDataConnector(Ingestor):
    """UN Data connector (download-first with limited API)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.portal_url = config.get("portal_url", "https://data.un.org/")
        self.live: bool = bool(config.get("live", False))
        self.mode: str = config.get("mode", "csv")  # csv | sdmx
        # CSV downloads specification list
        # { url, variable, unit, date_col, value_col, entity_col?, entity?, date_format? }
        self.downloads: List[Dict[str, Any]] = config.get("downloads", [])
        # SDMX queries
        # { dataset, filter, variable, unit, time_param? }
        self.sdmx_queries: List[Dict[str, Any]] = config.get("sdmx_queries", [])
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "population", "market": "demographics", "product": "population", "unit": "people", "update_freq": "annual"},
                {"name": "life_expectancy", "market": "demographics", "product": "life_exp_years", "unit": "years", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "csv_download": "https://data.un.org/Handlers/DownloadHandler.ashx?...",
                "sdmx_generic": "https://unstats.un.org/SDMX/v1/data/{DATASET}/{FILTER}?format=csv",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": "WLD", "variable": "population", "value": 7_900_000_000.0, "unit": "people"}
            yield {"timestamp": now, "entity": "WLD", "variable": "life_exp_years", "value": 73.2, "unit": "years"}
            return

        if self.mode == "csv" and self.downloads:
            for spec in self.downloads:
                yield from self._fetch_csv(spec)
        elif self.mode == "sdmx" and self.sdmx_queries:
            for q in self.sdmx_queries:
                yield from self._fetch_sdmx(q)
        else:
            logger.warning("UNData live mode configured but no downloads/sdmx_queries provided")

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WLD")
        variable = raw.get("variable", "population")
        instrument = f"UN.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} UN Data events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"UN Data checkpoint saved: {state}")

    # ---- Helpers ----
    def _fetch_csv(self, spec: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        url = spec.get("url")
        if not url:
            return
        variable = spec.get("variable", "un_series")
        unit = spec.get("unit", "unit")
        date_col = spec.get("date_col", "TIME")
        value_col = spec.get("value_col", "Value")
        entity_col = spec.get("entity_col")
        entity_fixed = spec.get("entity", "WLD")
        date_format = spec.get("date_format")  # e.g., %Y or %Y-%m
        encoding = spec.get("encoding", "utf-8")
        delimiter = spec.get("delimiter", ",")

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            content = resp.content.decode(encoding, errors="ignore")
        except Exception as e:
            logger.error(f"UN CSV download failed: {e} ({url})")
            return

        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        for row in reader:
            date_str = row.get(date_col)
            val = row.get(value_col)
            if not date_str or val in (None, ""):
                continue
            try:
                if date_format:
                    ts = datetime.strptime(date_str, date_format)
                else:
                    ts = datetime.fromisoformat(date_str)
            except Exception:
                try:
                    ts = datetime.strptime(date_str[:4], "%Y")
                except Exception:
                    continue
            entity = row.get(entity_col) if entity_col else entity_fixed
            try:
                value = float(val)
            except Exception:
                continue
            yield {
                "timestamp": ts.isoformat(),
                "entity": entity or "WLD",
                "variable": variable,
                "value": value,
                "unit": unit,
            }

    def _fetch_sdmx(self, q: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        dataset = q.get("dataset")
        flt = q.get("filter")
        variable = q.get("variable", dataset or "un_series")
        unit = q.get("unit", "unit")
        time_param = q.get("time_param")
        if not dataset or not flt:
            return
        base = "https://unstats.un.org/SDMX/v1/data"
        url = f"{base}/{dataset}/{flt}?format=csv"
        if time_param:
            url += f"&{time_param}"
        spec = {
            "url": url,
            "variable": variable,
            "unit": unit,
            "date_col": "TIME",
            "value_col": "Value",
            "entity_col": "REF_AREA"  # common SDMX column for region/country code
        }
        yield from self._fetch_csv(spec)


if __name__ == "__main__":
    connector = UNDataConnector({
        "source_id": "un_data",
        "live": False,
    })
    connector.run()
