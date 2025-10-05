"""
AESO (Alberta Electric System Operator) Connector

Overview
--------
Ingests Alberta electricity market data including Pool Price (SMP), Alberta
Internal Load (AIL), and intertie flows. Supports live AESO API integration
with configurable auth or generates mock data for development.

Auth Options
------------
- Authorization: ``Bearer <token>`` (preferred JWT/token)
- ``x-api-key: <key>`` (alternative header if key is not a bearer token)

Operational Notes
-----------------
- Enable live calls per‑endpoint via config flags (e.g., ``use_live_pool``) and
  fall back to mocks if upstream is unavailable.

 Data Flow
 ---------
 AESO API (or mocks) → normalize per product (SMP/AIL/flows) → canonical events → Kafka

 Configuration
 -------------
 - `api_base` and per-endpoint toggles for live vs. mock.
 - `kafka.topic`/`kafka.bootstrap_servers` for emission.
 - Optional auth header: `Authorization: Bearer <token>` or `x-api-key`.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator, Dict, Any
import time

import requests
from kafka import KafkaProducer
import json
import os

from .base import Ingestor

logger = logging.getLogger(__name__)


class AESOConnector(Ingestor):
    """
    AESO Alberta electricity market data connector.

    Responsibilities
    - Pull Pool Price (energy), AIL (load), Intertie (transmission), Generation mix
    - Support live AESO API calls for Pool/AIL with bearer/x-api-key auth
    - Provide graceful fallback to mocks when live API is disabled/unavailable
    - Map heterogeneous payloads to the platform’s canonical event schema

    Configuration
    - api_base: AESO API base (default https://api.aeso.ca/report/v1)
    - use_live_pool, use_live_ail: toggles for live endpoints (Pool, AIL)
    - pool_price_endpoint, ail_endpoint: relative paths joined to api_base
    - bearer_token or api_key (+ api_key_header): authentication
    - timeout_seconds, max_retries, retry_backoff_base: HTTP behavior
    - kafka_topic, kafka_bootstrap: emission settings
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Live API configuration (defaults are safe; override in config)
        self.api_base = config.get("api_base", "https://api.aeso.ca/report/v1")
        self.data_type = config.get("data_type", "POOL")  # POOL, AIL, INTERTIE, GENERATION
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
        # Live API toggles and endpoints
        self.use_live_pool = bool(config.get("use_live_pool", False))
        self.use_live_ail = bool(config.get("use_live_ail", False))
        # Endpoint paths (joined to api_base); override as needed
        self.pool_price_endpoint = config.get("pool_price_endpoint", "/price/poolPrice")
        self.ail_endpoint = config.get("ail_endpoint", "/load/albertaInternalLoad")
        # Auth options: prefer bearer token; fallback to x-api-key if provided
        self.bearer_token = config.get("bearer_token") or os.getenv("AESO_BEARER_TOKEN")
        self.api_key = config.get("api_key") or os.getenv("AESO_API_KEY")
        self.api_key_header = config.get("api_key_header", "x-api-key")
        # Networking
        self.timeout_seconds: int = int(config.get("timeout_seconds", 30))
        self.max_retries: int = int(config.get("max_retries", 3))
        self.retry_backoff_base: float = float(config.get("retry_backoff_base", 1.0))
    
    def discover(self) -> Dict[str, Any]:
        """Discover AESO data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "pool_price",
                    "market": "power",
                    "product": "energy",
                    "description": "Alberta Pool Price (System Marginal Price)",
                    "update_freq": "hourly",
                    "currency": "CAD",
                },
                {
                    "name": "ail",
                    "market": "power",
                    "product": "load",
                    "description": "Alberta Internal Load",
                    "update_freq": "realtime",
                },
                {
                    "name": "intertie",
                    "market": "power",
                    "product": "transmission",
                    "description": "Intertie flows with BC and Saskatchewan",
                    "update_freq": "realtime",
                },
                {
                    "name": "generation_mix",
                    "market": "power",
                    "product": "generation",
                    "description": "Generation by fuel type",
                    "update_freq": "realtime",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull data from AESO API."""
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching AESO {self.data_type} data since {last_time}")
        
        if self.data_type == "POOL":
            if self.use_live_pool:
                yield from self._fetch_pool_price_live()
            else:
                yield from self._fetch_pool_price()
        elif self.data_type == "AIL":
            if self.use_live_ail:
                yield from self._fetch_alberta_internal_load_live()
            else:
                yield from self._fetch_alberta_internal_load()
        elif self.data_type == "INTERTIE":
            yield from self._fetch_intertie()
        elif self.data_type == "GENERATION":
            yield from self._fetch_generation_mix()
    
    def _fetch_pool_price(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Alberta Pool Price (mock generator).
        
        The pool price is the system marginal price (SMP) in Alberta's
        energy-only market.
        """
        now = datetime.utcnow()
        
        # Generate pool price for each hour
        for hour in range(24):
            hour_ending = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Mock Pool Price (typically CAD $20-150/MWh, can spike higher)
            base_price = 55.0
            
            # Alberta load shape (peak in late afternoon/evening)
            if 16 <= hour <= 20:
                hourly_factor = 1.6  # Evening peak
            elif 7 <= hour <= 15 or 21 <= hour <= 23:
                hourly_factor = 1.3  # Shoulder
            else:
                hourly_factor = 0.7  # Off-peak
            
            # Add randomness and occasional spikes
            pool_price = base_price * hourly_factor + (hash(str(hour)) % 20) - 10
            
            # Occasional price spikes (5% chance)
            if hash(str(hour) + str(now.day)) % 20 == 0:
                pool_price *= 3.0  # Spike during tight supply
            
            pool_price = max(0, pool_price)  # Price floor at $0
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "AESO",
                "price_type": "POOL",
                "price": pool_price,
                "currency": "CAD",
                "hour_ending": hour_ending.isoformat(),
            }

    def _fetch_pool_price_live(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Pool Price from AESO live API (JSON).

        The API surface may evolve; this parser is defensive:
        - Detects common wrapper keys (data/records/items/rows/series)
        - Attempts to resolve timestamp and price fields via heuristics
        - Falls back to mocks if body is empty or unrecognized
        """
        url = self._join(self.api_base, self.pool_price_endpoint)
        data = self._get_json(url)
        if not data:
            logger.warning("AESO pool price live API returned no data; falling back to mock")
            yield from self._fetch_pool_price()
            return

        # Try common shapes: list of records or nested under a key
        records = self._extract_list(data)
        if not records:
            logger.warning("AESO pool price response unrecognized shape; using mock")
            yield from self._fetch_pool_price()
            return

        for rec in records:
            try:
                # Attempt to resolve timestamp and price fields across common variants
                ts = self._resolve_timestamp(rec, default=datetime.now(timezone.utc))
                price = self._resolve_number(rec, [
                    "pool_price", "poolPrice", "price", "smp", "SMP"
                ])
                if price is None:
                    continue
                yield {
                    "timestamp": ts.isoformat(),
                    "market": "AESO",
                    "price_type": "POOL",
                    "price": float(price),
                    "currency": "CAD",
                    "hour_ending": ts.replace(minute=0, second=0, microsecond=0).isoformat(),
                }
            except Exception as ex:
                logger.debug(f"Skip malformed AESO pool record: {ex}")
    
    def _fetch_alberta_internal_load(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Alberta Internal Load (AIL) (mock generator).
        
        Real-time electricity demand in Alberta.
        """
        now = datetime.utcnow()
        
        # Mock AIL (typically 7,000-12,000 MW)
        base_load = 9500  # MW
        
        # Hourly pattern
        hour = now.hour
        if 16 <= hour <= 20:
            hourly_factor = 1.25  # Evening peak
        elif 7 <= hour <= 15:
            hourly_factor = 1.15  # Daytime
        else:
            hourly_factor = 0.85  # Night
        
        # Seasonal adjustment
        month = now.month
        if month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.2  # AC load
        elif month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.4  # Heating load (Alberta is cold!)
        else:
            seasonal_factor = 1.0
        
        ail = base_load * hourly_factor * seasonal_factor
        ail += (hash(str(now.minute)) % 400) - 200
        
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "AESO",
                "metric": "AIL",
                "load_mw": ail,
            }

    def _fetch_alberta_internal_load_live(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch AIL from AESO live API (JSON).

        Similar to Pool Price handler, favors resilience over brittleness
        by inspecting multiple candidate field names for timestamps/values.
        """
        url = self._join(self.api_base, self.ail_endpoint)
        data = self._get_json(url)
        if not data:
            logger.warning("AESO AIL live API returned no data; falling back to mock")
            yield from self._fetch_alberta_internal_load()
            return

        records = self._extract_list(data)
        if not records:
            logger.warning("AESO AIL response unrecognized shape; using mock")
            yield from self._fetch_alberta_internal_load()
            return

        for rec in records:
            try:
                ts = self._resolve_timestamp(rec, default=datetime.now(timezone.utc))
                load = self._resolve_number(rec, [
                    "ail", "load_mw", "alberta_internal_load", "load"
                ])
                if load is None:
                    continue
                yield {
                    "timestamp": ts.isoformat(),
                    "market": "AESO",
                    "metric": "AIL",
                    "load_mw": float(load),
                }
            except Exception as ex:
                logger.debug(f"Skip malformed AESO AIL record: {ex}")
    
    def _fetch_intertie(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch intertie flows.
        
        Alberta has interties with:
        - British Columbia (BC Hydro)
        - Saskatchewan (SaskPower)
        - Montana (limited)
        """
        interties = [
            {
                "name": "AESO-BC",
                "capacity_import": 1000,
                "capacity_export": 1000,
                "typical_flow": 400,
            },
            {
                "name": "AESO-SASK",
                "capacity_import": 300,
                "capacity_export": 300,
                "typical_flow": 150,
            },
            {
                "name": "AESO-MONTANA",
                "capacity_import": 150,
                "capacity_export": 150,
                "typical_flow": 50,
            },
        ]
        
        now = datetime.utcnow()
        
        for intertie in interties:
            # Mock flow (MW)
            # Positive = import to Alberta, Negative = export from Alberta
            base_flow = intertie["typical_flow"]
            flow_variation = (hash(intertie["name"] + str(now.minute)) % 150) - 75
            flow = base_flow + flow_variation
            
            # Alberta often imports from BC (hydro)
            if "BC" in intertie["name"]:
                flow = abs(flow)  # Usually importing
            
            # Determine direction
            if flow > 0:
                direction = "IMPORT"
                actual_flow = min(flow, intertie["capacity_import"])
            else:
                direction = "EXPORT"
                actual_flow = max(flow, -intertie["capacity_export"])
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "AESO",
                "intertie": intertie["name"],
                "flow_mw": actual_flow,
                "direction": direction,
                "capacity_import_mw": intertie["capacity_import"],
                "capacity_export_mw": intertie["capacity_export"],
            }
    
    def _fetch_generation_mix(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch generation mix by fuel type.
        
        Alberta has diverse generation:
        - Coal (declining)
        - Natural Gas (growing)
        - Wind (significant and growing)
        - Hydro
        - Solar (small but growing)
        """
        now = datetime.utcnow()
        
        # Mock generation by fuel type (MW)
        fuel_types = [
            {"fuel": "GAS", "capacity": 6000, "typical_cf": 0.60},
            {"fuel": "COAL", "capacity": 2500, "typical_cf": 0.70},
            {"fuel": "WIND", "capacity": 4000, "typical_cf": 0.35},
            {"fuel": "HYDRO", "capacity": 900, "typical_cf": 0.50},
            {"fuel": "SOLAR", "capacity": 400, "typical_cf": 0.20},
            {"fuel": "OTHER", "capacity": 200, "typical_cf": 0.80},
        ]
        
        total_generation = 0
        
        for fuel in fuel_types:
            # Calculate generation
            if fuel["fuel"] == "WIND":
                # Wind varies more
                cf = fuel["typical_cf"] + (hash(str(now.hour)) % 30) / 100 - 0.15
            elif fuel["fuel"] == "SOLAR":
                # Solar depends on time of day
                if 6 <= now.hour <= 18:
                    cf = 0.6 * (1 - abs(now.hour - 12) / 12)
                else:
                    cf = 0.0
            else:
                cf = fuel["typical_cf"] + (hash(fuel["fuel"]) % 10) / 100 - 0.05
            
            cf = max(0, min(1, cf))
            generation_mw = fuel["capacity"] * cf
            total_generation += generation_mw
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "AESO",
                "fuel_type": fuel["fuel"],
                "generation_mw": generation_mw,
                "capacity_mw": fuel["capacity"],
                "capacity_factor": cf,
            }
        
        # Total generation summary
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "AESO",
            "metric": "TOTAL_GENERATION",
            "generation_mw": total_generation,
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map AESO format to canonical schema.

        Product mapping
        - energy: Pool Price (value in CAD/MWh)
        - load: AIL (MW)
        - transmission: intertie net flow (MW), sign reflects direction
        - generation: per-fuel MW and total generation snapshot
        """
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product, location, and value
        if "price_type" in raw and raw["price_type"] == "POOL":
            product = "energy"
            location = "AESO.ALBERTA"
            value = raw["price"]
        elif "metric" in raw and raw["metric"] == "AIL":
            product = "load"
            location = "AESO.ALBERTA"
            value = raw["load_mw"]
        elif "intertie" in raw:
            product = "transmission"
            location = raw["intertie"]
            value = raw["flow_mw"]
        elif "fuel_type" in raw:
            product = "generation"
            location = f"AESO.GEN.{raw['fuel_type']}"
            value = raw["generation_mw"]
        elif "metric" in raw and raw["metric"] == "TOTAL_GENERATION":
            product = "generation"
            location = "AESO.ALBERTA"
            value = raw["generation_mw"]
        else:
            product = "unknown"
            location = "AESO.UNKNOWN"
            value = 0
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "trade",
            "value": float(value),
            "volume": None,
            "currency": raw.get("currency", "CAD"),
            "unit": "MWh" if product == "energy" else "MW",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }

    # --- helpers ---------------------------------------------------------
    def _auth_headers(self) -> Dict[str, str]:
        """Build auth headers for AESO API (bearer token or x-api-key)."""
        headers: Dict[str, str] = {"User-Agent": "254Carbon-Platform/1.0"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if self.api_key:
            headers[self.api_key_header] = str(self.api_key)
        return headers

    def _join(self, base: str, path: str) -> str:
        """Join base URL and path, handling slashes cleanly."""
        if base.endswith("/") and path.startswith("/"):
            return base[:-1] + path
        if not base.endswith("/") and not path.startswith("/"):
            return base + "/" + path
        return base + path

    def _get_json(self, url: str) -> Any:
        """GET a JSON payload with retry/backoff and early-exit on 4xx auth errors."""
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.get(url, headers=self._auth_headers(), timeout=self.timeout_seconds)
                if resp.status_code == 200:
                    return resp.json()
                if attempt > self.max_retries or resp.status_code in (400, 401, 403):
                    logger.error(f"AESO API HTTP {resp.status_code} for {url}")
                    return None
            except Exception as ex:
                if attempt > self.max_retries:
                    logger.error(f"AESO API error after retries: {ex}")
                    return None
            time.sleep(self.retry_backoff_base * (2 ** (attempt - 1)))

    def _extract_list(self, data: Any) -> Any:
        """Extract list-like payload from common wrapper shapes; else return None."""
        if data is None:
            return None
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # common wrappers
            for k in ("data", "return", "items", "records", "rows", "series"):
                if k in data and isinstance(data[k], list):
                    return data[k]
        return None

    def _resolve_timestamp(self, rec: Dict[str, Any], default: datetime) -> datetime:
        """Resolve timestamp from heterogeneous field names and formats (epoch or ISO)."""
        candidates = [
            "timestamp", "time", "effectiveAt", "effective_time", "begin", "start", "hour_ending"
        ]
        for k in candidates:
            v = rec.get(k)
            if not v:
                continue
            try:
                # Try epoch ms/seconds
                if isinstance(v, (int, float)):
                    # Heuristic: ms vs s
                    if v > 1e12:
                        return datetime.fromtimestamp(v / 1000, tz=timezone.utc)
                    return datetime.fromtimestamp(v, tz=timezone.utc)
                # ISO8601
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except Exception:
                continue
        return default

    def _resolve_number(self, rec: Dict[str, Any], keys: list[str]) -> Any:
        """Resolve first numeric field among preferred keys or any numeric value in record."""
        for k in keys:
            v = rec.get(k)
            if v is None or v == "":
                continue
            try:
                return float(v)
            except Exception:
                continue
        # try to find the first numeric value
        for v in rec.values():
            try:
                return float(v)
            except Exception:
                continue
        return None
    
    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """Emit events to Kafka."""
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
        
        self.producer.flush()
        logger.info(f"Emitted {count} AESO events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"AESO checkpoint saved: {state}")


if __name__ == "__main__":
    # Test AESO connector
    configs = [
        {
            "source_id": "aeso_pool",
            "data_type": "POOL",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "aeso_ail",
            "data_type": "AIL",
            "kafka_topic": "power.load.v1",
        },
        {
            "source_id": "aeso_intertie",
            "data_type": "INTERTIE",
            "kafka_topic": "power.transmission.v1",
        },
        {
            "source_id": "aeso_generation",
            "data_type": "GENERATION",
            "kafka_topic": "power.generation.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = AESOConnector(config)
        connector.run()
