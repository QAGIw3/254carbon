"""
US Census Bureau APIs Connector

Coverage: US population, housing, economics, geography.
Portal: https://www.census.gov/data/developers/data-sets.html

Live mode queries the Census API (e.g., decennial P.L. 94-171, ACS) using
configurable dataset paths, variables, and geographic filters, then maps to
canonical series such as population and housing_units.
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


class USCensusConnector(Ingestor):
    """US Census connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # US Census API base: https://api.census.gov/data
        # Example: /2020/dec/pl?get=NAME,P1_001N&for=state:*&key=...
        self.api_base = config.get("api_base", "https://api.census.gov/data")
        self.dataset = config.get("dataset", "2020/dec/pl")
        # Live mode toggle
        self.live: bool = bool(config.get("live", False))
        # Variables to get (Census 'get' parameter)
        self.variables: List[str] = config.get("variables", ["NAME", "P1_001N"])  # total pop
        # Geography filters
        # e.g., geo_for='state:*' or 'county:*', geo_in='state:06'
        self.geo_for: str = config.get("geo_for", "state:*")
        self.geo_in: Optional[str] = config.get("geo_in")
        self.api_key: Optional[str] = config.get("api_key")
        # Optional timestamp year field; else use provided year or current
        self.timestamp_year_field: Optional[str] = config.get("timestamp_year_field")
        self.year: Optional[int] = config.get("year")
        # Aliases map raw variable codes to (variable, unit)
        self.aliases: Dict[str, Tuple[str, str]] = config.get("aliases", {
            "P1_001N": ("population", "people"),
            "B25001_001E": ("housing_units", "units"),
        })
        # Build entity label from this field if present (e.g., NAME). Otherwise join geo ids
        self.entity_field: Optional[str] = config.get("entity_field", "NAME")
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "population", "market": "demographics", "product": "population", "unit": "people", "update_freq": "annual"},
                {"name": "housing_units", "market": "demographics", "product": "housing_units", "unit": "units", "update_freq": "annual"},
            ],
            "endpoint_examples": {
                "population_state": (
                    "GET {base}/2020/dec/pl?get=NAME,P1_001N&for=state:*&key=..."
                ),
                "acs_households": (
                    "GET {base}/2022/acs/acs1?get=NAME,B25001_001E&for=state:*&key=..."
                ),
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": "US", "variable": "population", "value": 331_000_000.0, "unit": "people"}
            yield {"timestamp": now, "entity": "US", "variable": "housing_units", "value": 140_000_000.0, "unit": "units"}
            return

        url = f"{self.api_base.rstrip('/')}/{self.dataset.lstrip('/')}"
        params: Dict[str, Any] = {
            "get": ",".join(self.variables),
            "for": self.geo_for,
        }
        if self.geo_in:
            params["in"] = self.geo_in
        if self.api_key:
            params["key"] = self.api_key

        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Census API request failed: {e}")
            return

        if not data or not isinstance(data, list) or len(data) < 2:
            return
        header = data[0]
        rows = data[1:]

        # Build indices for quick lookups
        idx: Dict[str, int] = {name: i for i, name in enumerate(header)}

        for row in rows:
            # Determine entity label
            entity = "US"
            if self.entity_field and self.entity_field in idx:
                try:
                    entity = row[idx[self.entity_field]]
                except Exception:
                    entity = "US"
            else:
                # fallback: use geo keys like state/county
                parts = []
                for key_name in ("state", "county", "tract", "block group"):
                    if key_name in idx:
                        parts.append(f"{key_name}:{row[idx[key_name]]}")
                if parts:
                    entity = ".".join(parts)

            # Determine timestamp
            ts_iso = self._resolve_timestamp(header, row)

            # Emit one message per numeric variable in 'variables'
            for var in self.variables:
                if var in (self.entity_field or "", "NAME"):
                    continue
                if var not in idx:
                    continue
                raw_val = row[idx[var]]
                try:
                    val = float(raw_val)
                except Exception:
                    continue
                var_name, unit = self.aliases.get(var, (var.lower(), "unit"))
                yield {
                    "timestamp": ts_iso,
                    "entity": entity,
                    "variable": var_name,
                    "value": float(val),
                    "unit": unit,
                }

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "US")
        variable = raw.get("variable", "population")
        instrument = f"CENSUS.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} US Census events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"US Census checkpoint saved: {state}")

    def _resolve_timestamp(self, header: List[str], row: List[str]) -> str:
        # Use configured year field, else 'YEAR' or current year
        if self.timestamp_year_field and self.timestamp_year_field in header:
            try:
                y = int(row[header.index(self.timestamp_year_field)])
                return f"{y}-01-01T00:00:00Z"
            except Exception:
                pass
        if "YEAR" in header:
            try:
                y = int(row[header.index("YEAR")])
                return f"{y}-01-01T00:00:00Z"
            except Exception:
                pass
        if self.year:
            return f"{int(self.year)}-01-01T00:00:00Z"
        return f"{datetime.utcnow().year}-01-01T00:00:00Z"


if __name__ == "__main__":
    connector = USCensusConnector({"source_id": "us_census"})
    connector.run()
