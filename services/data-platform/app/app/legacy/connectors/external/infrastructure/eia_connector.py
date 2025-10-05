"""
US EIA Open Data Connector

Coverage: US energy production, consumption, prices, electricity, fuels.
Docs: https://www.eia.gov/opendata/

This scaffold uses safe mocked data and exposes clear configuration for
real API integration (API key, dataset, filters). It emits canonical
events suitable for fundamentals ingestion.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from kafka import KafkaProducer
import json
import requests
from urllib.parse import urlencode

from ...base import Ingestor


# Default query specifications to ensure refining and renewables analytics
# have coverage even when Airflow/connector configuration does not override
# them yet. They can be overridden via config['queries'].
DEFAULT_REFINING_AND_RENEWABLE_QUERIES: List[Dict[str, Any]] = [
    {
        "path": "petroleum/pri/gnd/data",
        "variable": "gasoline_price",
        "unit": "USD/gal",
        "frequency": "weekly",
        "facets": {"product": ["RBOB"], "area": ["USA"]},
        "entity_fields": ["area"],
    },
    {
        "path": "petroleum/pri/gnd/data",
        "variable": "diesel_price",
        "unit": "USD/gal",
        "frequency": "weekly",
        "facets": {"product": ["ULSD"], "area": ["USA"]},
        "entity_fields": ["area"],
    },
    {
        "path": "petroleum/pri/jet/data",
        "variable": "jet_price",
        "unit": "USD/gal",
        "frequency": "weekly",
        "facets": {"area": ["USA"]},
        "entity_fields": ["area"],
    },
    {
        "path": "petroleum/pri/hep/data",
        "variable": "heating_oil_price",
        "unit": "USD/gal",
        "frequency": "weekly",
        "facets": {"area": ["USA"]},
        "entity_fields": ["area"],
    },
    {
        "path": "petroleum/pri/spot/data",
        "variable": "wti_crude_price",
        "unit": "USD/bbl",
        "frequency": "daily",
        "facets": {"series": ["RWTCL"]},
        "entity_fields": ["series"],
    },
    {
        "path": "petroleum/pri/spot/data",
        "variable": "brent_crude_price",
        "unit": "USD/bbl",
        "frequency": "daily",
        "facets": {"series": ["RBRTE"]},
        "entity_fields": ["series"],
    },
    {
        "path": "petroleum/pnp/hist/data",
        "variable": "gasoline_demand",
        "unit": "kbbl/d",
        "frequency": "monthly",
        "facets": {"series": ["MGTUPUS2"], "process": ["FGUL"]},
        "entity_fields": ["series"],
    },
    {
        "path": "petroleum/pnp/hist/data",
        "variable": "diesel_demand",
        "unit": "kbbl/d",
        "frequency": "monthly",
        "facets": {"series": ["MDTUPUS2"], "process": ["_No_" ]},
        "entity_fields": ["series"],
    },
    {
        "path": "petroleum/pnp/hist/data",
        "variable": "jet_fuel_demand",
        "unit": "kbbl/d",
        "frequency": "monthly",
        "facets": {"series": ["MKTJEPUS1"]},
        "entity_fields": ["series"],
    },
]

# ---------------------------------------------------------------------------
# Canonical mappings for refinery and renewables analytics ingestion.
# ---------------------------------------------------------------------------

# Price variables that should land in ch.market_price_ticks with standardized
# instrument identifiers and units.
PRICE_VARIABLE_MAP: Dict[str, Dict[str, str]] = {
    "gasoline_price": {"instrument_id": "OIL.RBOB", "unit": "USD/gal", "market": "refined_products"},
    "diesel_price": {"instrument_id": "OIL.ULSD", "unit": "USD/gal", "market": "refined_products"},
    "jet_price": {"instrument_id": "OIL.JET", "unit": "USD/gal", "market": "refined_products"},
    "heating_oil_price": {"instrument_id": "OIL.HEATOIL", "unit": "USD/gal", "market": "refined_products"},
    "wti_crude_price": {"instrument_id": "OIL.WTI", "unit": "USD/bbl", "market": "crude_oil"},
    "brent_crude_price": {"instrument_id": "OIL.BRENT", "unit": "USD/bbl", "market": "crude_oil"},
    "biodiesel_price": {"instrument_id": "BIO.BIODIESEL", "unit": "USD/gal", "market": "biofuels"},
}

# Fundamentals variables that should land in ch.fundamentals_series. Entity
# identifiers are keyed by geographic region (e.g., US, PADD1).
FUNDAMENTAL_VARIABLE_MAP: Dict[str, Dict[str, str]] = {
    "gasoline_demand": {"unit": "kbbl/d", "market": "transport_fuels"},
    "diesel_demand": {"unit": "kbbl/d", "market": "transport_fuels"},
    "jet_fuel_demand": {"unit": "kbbl/d", "market": "transport_fuels"},
    "ethanol_blending_rate": {"unit": "pct", "market": "transport_fuels"},
    "biodiesel_blending_rate": {"unit": "pct", "market": "transport_fuels"},
    "gdp_proxy_index": {"unit": "index", "market": "macro"},
    "ev_adoption_index": {"unit": "index", "market": "transport_fuels"},
    "padd_refinery_utilization": {"unit": "pct", "market": "refining"},
}

logger = logging.getLogger(__name__)


class EIAOpenDataConnector(Ingestor):
    """US EIA Open Data connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # EIA v2 API base. Typical GET:
        # https://api.eia.gov/v2/{category}/{dataset}/data/?api_key=...&start=YYYY-MM&end=YYYY-MM&data[]=value&facets[state][]=CA
        self.api_base = config.get("api_base", "https://api.eia.gov/v2")
        self.api_key = config.get("api_key")
        # Live mode toggle; if true, use v2 API
        self.live: bool = bool(config.get("live", False))
        # Simple mode: list of dataset paths to pull with minimal params
        self.datasets: List[str] = config.get("datasets", [])
        # Advanced mode: list of query specifications for multiple variables
        # Example item:
        # {
        #   'path': 'electricity/retail-sales',
        #   'variable': 'retail_sales',
        #   'unit': 'GWh',
        #   'start': '2020-01', 'end': '2025-01', 'frequency': 'monthly',
        #   'facets': {'state': ['CA']},
        #   'entity_fields': ['state']
        # }
        self.queries: List[Dict[str, Any]] = config.get("queries", [])
        if config.get("enable_default_queries", True) and not self.datasets and not self.queries:
            self.queries = DEFAULT_REFINING_AND_RENEWABLE_QUERIES.copy()
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "electricity_sales",
                    "market": "infra",
                    "product": "electricity",
                    "variables": ["retail_sales"],
                    "update_freq": "monthly",
                },
                {
                    "name": "fuel_prices",
                    "market": "infra",
                    "product": "fuels",
                    "variables": ["gasoline_price", "diesel_price"],
                    "update_freq": "weekly",
                },
            ],
            "endpoint_examples": {
                "datasets": "GET https://api.eia.gov/v2/datasets?api_key=...",
                "electricity_retail_sales": (
                    "GET {base}/electricity/retail-sales/data/?api_key=..."
                    "&start=2020-01&end=2025-01&data[]=value&facets[state][]=CA"
                ),
                "weekly_fuel_prices": (
                    "GET {base}/petroleum/pri/gnd/data/?api_key=..."
                    "&frequency=weekly&data[]=value&facets[product][]=MGE&facets[area][]=PADD"
                ),
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull from EIA v2 API or emit mock data when live is disabled."""
        if not self.live:
            now = datetime.utcnow().isoformat()
            # Electricity retail sales (mock, GWh)
            yield {
                "timestamp": now,
                "domain": "electricity",
                "entity": "US",
                "variable": "retail_sales",
                "value": 387000.0,
                "unit": "GWh",
            }
            # Gasoline and diesel prices (USD/gal)
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "gasoline_price",
                "value": 3.45,
                "unit": "USD/gal",
            }
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "diesel_price",
                "value": 4.07,
                "unit": "USD/gal",
            }
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "jet_price",
                "value": 3.95,
                "unit": "USD/gal",
            }
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "heating_oil_price",
                "value": 3.60,
                "unit": "USD/gal",
            }
            yield {
                "timestamp": now,
                "domain": "crude",
                "entity": "US",
                "variable": "wti_crude_price",
                "value": 78.40,
                "unit": "USD/bbl",
            }
            yield {
                "timestamp": now,
                "domain": "crude",
                "entity": "ICE",
                "variable": "brent_crude_price",
                "value": 82.15,
                "unit": "USD/bbl",
            }
            yield {
                "timestamp": now,
                "domain": "biofuels",
                "entity": "US",
                "variable": "biodiesel_price",
                "value": 4.55,
                "unit": "USD/gal",
            }
            # Transport fundamentals (mock, consistent units)
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "gasoline_demand",
                "value": 8300.0,
                "unit": "kbbl/d",
            }
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "diesel_demand",
                "value": 4300.0,
                "unit": "kbbl/d",
            }
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "jet_fuel_demand",
                "value": 1500.0,
                "unit": "kbbl/d",
            }
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "ethanol_blending_rate",
                "value": 10.2,
                "unit": "pct",
            }
            yield {
                "timestamp": now,
                "domain": "fuels",
                "entity": "US",
                "variable": "biodiesel_blending_rate",
                "value": 4.8,
                "unit": "pct",
            }
            yield {
                "timestamp": now,
                "domain": "macro",
                "entity": "US",
                "variable": "gdp_proxy_index",
                "value": 101.3,
                "unit": "index",
            }
            yield {
                "timestamp": now,
                "domain": "transport",
                "entity": "US",
                "variable": "ev_adoption_index",
                "value": 58.5,
                "unit": "index",
            }
            yield {
                "timestamp": now,
                "domain": "refining",
                "entity": "PADD3",
                "variable": "padd_refinery_utilization",
                "value": 89.2,
                "unit": "pct",
            }
            return

        if not self.api_key:
            logger.error("EIA live mode requires api_key in config")
            return

        # Prefer advanced query specs; fall back to bare datasets list
        if self.queries:
            for q in self.queries:
                yield from self._fetch_query(q)
        elif self.datasets:
            for path in self.datasets:
                q = {
                    "path": path,
                    "variable": path.split("/")[-1].replace("-", "_"),
                    "unit": "unit",
                }
                yield from self._fetch_query(q)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        domain = raw.get("domain", "electricity")
        entity = raw.get("entity", "US")
        variable = raw.get("variable", domain)
        value = float(raw.get("value", 0.0))

        # Price payload → standardized instrument_id in ch.market_price_ticks
        if variable in PRICE_VARIABLE_MAP:
            price_cfg = PRICE_VARIABLE_MAP[variable]
            instrument = price_cfg["instrument_id"]
            unit = price_cfg.get("unit", raw.get("unit", "unit"))
            market = price_cfg.get("market", domain)
            return {
                "event_time_utc": int(ts.timestamp() * 1000),
                "market": market,
                "product": variable,
                "instrument_id": instrument,
                "location_code": entity,
                "price_type": raw.get("price_type", "spot"),
                "value": value,
                "volume": raw.get("volume"),
                "currency": "USD",
                "unit": unit,
                "source": self.source_id,
                "seq": int(time.time() * 1000000),
            }

        # Fundamentals payload → ch.fundamentals_series keyed by entity + variable
        if variable in FUNDAMENTAL_VARIABLE_MAP:
            fund_cfg = FUNDAMENTAL_VARIABLE_MAP[variable]
            unit = fund_cfg.get("unit", raw.get("unit", "unit"))
            market = fund_cfg.get("market", domain)
            return {
                "event_time_utc": int(ts.timestamp() * 1000),
                "market": market,
                "product": variable,
                "instrument_id": entity,
                "location_code": entity,
                "price_type": "observation",
                "value": value,
                "volume": raw.get("volume"),
                "currency": raw.get("currency", "USD"),
                "unit": unit,
                "source": self.source_id,
                "seq": int(time.time() * 1000000),
            }

        # Fallback for legacy datasets: retain original namespaced instrument id
        instrument = f"EIA.{entity}.{domain.upper()}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": domain,
            "product": variable,
            "instrument_id": instrument,
            "location_code": instrument,
            "price_type": "observation",
            "value": value,
            "volume": raw.get("volume"),
            "currency": raw.get("currency", "USD"),
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
        logger.info(f"Emitted {count} EIA events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"EIA checkpoint saved: {state}")

    # --- helpers ---
    def _fetch_query(self, q: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Fetch a single query spec and yield unified raw events.

        q keys:
          - path: str, required (e.g., 'electricity/retail-sales')
          - variable: str, required (e.g., 'retail_sales')
          - unit: str, optional
          - start/end: optional period strings
          - frequency: optional ('annual','monthly','weekly','daily')
          - facets: dict[str, list[str]]
          - entity_fields: list[str] for building entity label from response (e.g., ['state'])
        """
        path = q.get("path")
        if not path:
            return
        variable = q.get("variable", path.split("/")[-1])
        unit = q.get("unit", "unit")
        start = q.get("start")
        end = q.get("end")
        frequency = q.get("frequency")
        facets: Dict[str, List[str]] = q.get("facets", {}) or {}
        entity_fields: List[str] = q.get("entity_fields", [])

        url = self._compose_url(path)
        params: Dict[str, Any] = {"api_key": self.api_key}
        # Data fields to return
        params.setdefault("data[]", "value")
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if frequency:
            params["frequency"] = frequency
        # Encode facets
        for k, vals in facets.items():
            for v in vals:
                params.setdefault(f"facets[{k}][]", v)

        offset = 0
        length = 5000
        while True:
            page_params = params.copy()
            page_params["offset"] = offset
            page_params["length"] = length
            try:
                resp = requests.get(url, params=page_params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as e:
                logger.error(f"EIA request failed for {path}: {e}")
                break

            data = (payload or {}).get("response", {}).get("data", [])
            if not data:
                break

            for row in data:
                period = row.get("period")
                value = row.get("value")
                if period is None or value is None:
                    continue
                ts_iso = self._parse_period_iso(period)
                entity = self._build_entity_label(row, entity_fields) or "US"
                domain = path.split("/", 1)[0] if "/" in path else "infra"
                yield {
                    "timestamp": ts_iso,
                    "domain": domain,
                    "entity": entity,
                    "variable": variable,
                    "value": float(value),
                    "unit": unit,
                }

            if len(data) < length:
                break
            offset += length

    def _compose_url(self, path: str) -> str:
        # Ensure trailing '/data' if missing
        if not path.endswith("/data"):
            if not path.endswith("/"):
                path = path + "/"
            path = path + "data"
        return f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"

    def _parse_period_iso(self, period: str) -> str:
        # Normalize EIA period to ISO date at midnight UTC
        p = period.strip()
        try:
            if len(p) == 4 and p.isdigit():  # YYYY
                return f"{p}-01-01T00:00:00Z"
            if len(p) == 7 and p[4] in "-":  # YYYY-MM
                return f"{p}-01T00:00:00Z"
            if len(p) == 6 and p.isdigit():  # YYYYMM
                return f"{p[0:4]}-{p[4:6]}-01T00:00:00Z"
            if len(p) == 8 and p.isdigit():  # YYYYMMDD
                return f"{p[0:4]}-{p[4:6]}-{p[6:8]}T00:00:00Z"
            # Already ISO-like
            if "T" in p or len(p) >= 10:
                # Keep date part
                return p if "T" in p else f"{p}T00:00:00Z"
        except Exception:
            pass
        # Fallback: now
        return datetime.utcnow().isoformat()

    def _build_entity_label(self, row: Dict[str, Any], fields: List[str]) -> Optional[str]:
        parts: List[str] = []
        for f in fields:
            val = row.get(f)
            if val is not None:
                parts.append(str(val))
        if parts:
            return ".".join(parts)
        # Common fields to try
        for f in ("state", "series", "area", "duoarea"):
            if row.get(f):
                return str(row[f])
        return None


if __name__ == "__main__":
    connector = EIAOpenDataConnector({"source_id": "eia_open_data"})
    connector.run()
