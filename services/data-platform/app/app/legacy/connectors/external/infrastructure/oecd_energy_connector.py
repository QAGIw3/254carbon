"""
OECD Energy Statistics Connector

Overview
--------
Pulls selected OECD energy datasets via CSV/Excel downloads or the SDMX API
where available. In development, emits deterministic sample indicators.

Data Flow
---------
OECD CSV/SDMX → normalize indicator rows → canonical fundamentals → Kafka

Configuration
-------------
- `live`: Toggle live downloads (default False, emit mocks).
- `mode`: `csv` (default) or `sdmx`.
- `downloads`: List of CSV specs: { url, variable, unit, date_col, value_col,
  country_col?, country?, date_format?, encoding?, delimiter? }.
- `sdmx_queries`: List of SDMX specs: { dataset, filter, variable, unit,
  time_param? }.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- Many OECD energy datasets are sourced from IEA and may be license-restricted;
  keep URLs and dataset selection compliant with your entitlements.
- CSV parsing attempts ISO or format-specific parsing, falling back to YYYY.
  Rows without a timestamp or value are skipped.
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


class OECDEnergyStatsConnector(Ingestor):
    """OECD energy statistics connector (download-first with optional SDMX)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.portal_url = config.get("portal_url", "https://data.oecd.org/energy.htm")
        # Live mode toggles
        self.live: bool = bool(config.get("live", False))
        self.mode: str = config.get("mode", "csv")  # csv | sdmx
        # CSV downloads configuration
        # downloads: [ { url, variable, unit, date_col, value_col, country_col?, country?, date_format? } ]
        self.downloads: List[Dict[str, Any]] = config.get("downloads", [])
        # SDMX configuration (optional)
        # sdmx_queries: [ { dataset, filter, variable, unit, time_param? } ]
        self.sdmx_queries: List[Dict[str, Any]] = config.get("sdmx_queries", [])
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        """Describe available streams and example endpoints."""
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "energy_balance", "market": "infra", "product": "energy_balance_ktoe", "unit": "ktoe", "update_freq": "annual"},
                {"name": "household_prices", "market": "infra", "product": "household_electricity_price", "unit": "USD/MWh", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "csv_download": "https://stats.oecd.org/some/public/csv/path.csv",
                "sdmx_generic": "https://stats.oecd.org/sdmx-json/data/{DATASET}/{FILTER}.json?contentType=csv",
                "note": "Many detailed energy datasets are IEA-sourced (licensed).",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Download and yield indicator rows, or emit mocks in dev mode."""
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": "OECD", "variable": "energy_balance_ktoe", "value": 1_250_000.0, "unit": "ktoe"}
            yield {"timestamp": now, "entity": "OECD", "variable": "household_electricity_price", "value": 210.0, "unit": "USD/MWh"}
            return

        if self.mode == "csv" and self.downloads:
            for spec in self.downloads:
                yield from self._fetch_csv(spec)
        elif self.mode == "sdmx" and self.sdmx_queries:
            for q in self.sdmx_queries:
                yield from self._fetch_sdmx(q)
        else:
            logger.warning("No live download configuration provided (downloads/sdmx_queries)")

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map OECD indicator rows to the canonical infrastructure format."""
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "OECD")
        variable = raw.get("variable", "energy_balance_ktoe")
        instrument = f"OECD.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} OECD Energy events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Persist last download state (simple assignment here)."""
        self.checkpoint_state = state
        logger.debug(f"OECD Energy checkpoint saved: {state}")

    # ---- Helpers ----
    def _fetch_csv(self, spec: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Download a CSV and yield unified indicator rows.

        Spec keys
        ---------
        url, variable, unit, date_col, value_col, country_col?, country?,
        date_format? (strftime), encoding?, delimiter?
        """
        url = spec.get("url")
        if not url:
            return
        variable = spec.get("variable", "oecd_energy")
        unit = spec.get("unit", "unit")
        date_col = spec.get("date_col", "TIME")
        value_col = spec.get("value_col", "Value")
        country_col = spec.get("country_col")
        country_fixed = spec.get("country")
        date_format = spec.get("date_format")  # e.g., %Y or %Y-%m
        encoding = spec.get("encoding", "utf-8")
        delimiter = spec.get("delimiter", ",")

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            content = resp.content.decode(encoding, errors="ignore")
        except Exception as e:
            logger.error(f"OECD CSV download failed: {e} ({url})")
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
                    # Attempt ISO or YYYY fallback
                    ts = datetime.fromisoformat(date_str)
            except Exception:
                # Fallback YYYY
                try:
                    ts = datetime.strptime(date_str[:4], "%Y")
                except Exception:
                    continue

            entity = country_fixed or row.get(country_col) or "OECD"
            yield {
                "timestamp": ts.isoformat(),
                "entity": entity,
                "variable": variable,
                "value": float(val),
                "unit": unit,
            }

    def _fetch_sdmx(self, q: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Fetch SDMX generic CSV for a dataset/filter pair and yield rows."""
        dataset = q.get("dataset")
        flt = q.get("filter")
        variable = q.get("variable", dataset or "oecd_energy")
        unit = q.get("unit", "unit")
        time_param = q.get("time_param")  # e.g., time=2020-2025
        if not dataset or not flt:
            return
        base = "https://stats.oecd.org/sdmx-json/data"
        url = f"{base}/{dataset}/{flt}?contentType=csv"
        if time_param:
            url += f"&{time_param}"
        spec = {
            "url": url,
            "variable": variable,
            "unit": unit,
            "date_col": "TIME",
            "value_col": "Value",
            "country_col": "LOCATION",
        }
        yield from self._fetch_csv(spec)


if __name__ == "__main__":
    connector = OECDEnergyStatsConnector({
        "source_id": "oecd_energy",
        "live": False,
    })
    connector.run()
