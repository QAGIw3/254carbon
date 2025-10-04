"""
GIE ALSI LNG Inventory Connector
---------------------------------

Fetches European LNG terminal transparency data from the Gas Infrastructure
Europe (GIE) Aggregated LNG Storage Inventory (ALSI) API.

Capabilities
~~~~~~~~~~~~
- Terminal, country, or EU-level granularity (configurable)
- Tracks LNG inventory, send-out, and ship arrivals
- Emits canonical infrastructure events (market.fundamentals → ClickHouse)
- Handles rate limits with retries/backoff
- Persists checkpoints via base Ingestor

API reference: https://alsi.gie.eu/
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import (
    InfrastructureConnector,
    LNGTerminal,
    GeoLocation,
    OperationalStatus,
    InfrastructureType
)
from ....base import CommodityType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LNGStorageRecord:
    """Normalized LNG storage observation from ALSI."""
    
    as_of: datetime
    entity: str
    entity_type: str  # terminal|country|eu
    inventory_gwh: Optional[float]
    inventory_mcm: Optional[float]
    fullness_pct: Optional[float]
    send_out_gwh: Optional[float]
    ship_arrivals: Optional[int]
    capacity_gwh: Optional[float]
    capacity_mcm: Optional[float]
    region: Optional[str]
    raw: Dict[str, Any]


class ALSILNGConnector(InfrastructureConnector):
    """Connector for the GIE Aggregated LNG Storage Inventory (ALSI)."""
    
    DEFAULT_API_BASE = "https://alsi.gie.eu/api"
    
    # Known LNG terminal locations (subset for initial implementation)
    TERMINAL_LOCATIONS = {
        # Spain
        "ES_BARCELONA": GeoLocation(41.3851, 2.1734),
        "ES_BILBAO": GeoLocation(43.3975, -3.0156),
        "ES_CARTAGENA": GeoLocation(37.5987, -0.9813),
        "ES_HUELVA": GeoLocation(37.2156, -6.9515),
        "ES_MUGARDOS": GeoLocation(43.4611, -8.2556),
        "ES_SAGUNTO": GeoLocation(39.6308, -0.2139),
        
        # France
        "FR_DUNKERQUE": GeoLocation(51.0169, 2.2053),
        "FR_FOS_CAVAOU": GeoLocation(43.3736, 4.8556),
        "FR_FOS_TONKIN": GeoLocation(43.3869, 4.8458),
        "FR_MONTOIR": GeoLocation(47.3105, -2.1669),
        
        # Italy
        "IT_LA_SPEZIA": GeoLocation(44.0947, 9.8472),
        "IT_LIVORNO": GeoLocation(43.5883, 10.3006),
        "IT_RAVENNA": GeoLocation(44.4847, 12.2472),
        
        # Netherlands
        "NL_GATE": GeoLocation(51.9456, 4.1194),
        
        # Belgium
        "BE_ZEEBRUGGE": GeoLocation(51.3322, 3.1967),
        
        # Poland
        "PL_SWINOUJSCIE": GeoLocation(53.9103, 14.2472),
        
        # Portugal
        "PT_SINES": GeoLocation(37.9569, -8.8642),
        
        # Greece
        "GR_REVITHOUSSA": GeoLocation(37.9375, 23.3961),
        
        # UK
        "UK_GRAIN": GeoLocation(51.4464, 0.7156),
        "UK_MILFORD_HAVEN": GeoLocation(51.7078, -5.0342),
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.GAS
        
        self.api_base = config.get("api_base", self.DEFAULT_API_BASE).rstrip("/")
        self.api_key = config.get("api_key")
        self.granularity = (config.get("granularity") or "terminal").lower()
        self.entities: Optional[List[str]] = config.get("entities")
        self.lookback_days = int(config.get("lookback_days", 7))
        self.include_rollups = bool(config.get("include_rollups", False))
        
        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")
        
        self.session = requests.Session()
        self._validate_config()
        self._initialize_terminals()
    
    def _initialize_terminals(self) -> None:
        """Initialize LNG terminal assets."""
        
        for terminal_code, location in self.TERMINAL_LOCATIONS.items():
            country = terminal_code.split("_")[0]
            terminal = LNGTerminal(
                asset_id=terminal_code,
                name=terminal_code.replace("_", " "),
                location=location,
                country=country,
                status=OperationalStatus.OPERATIONAL,
                storage_capacity_gwh=100.0,  # Default, will be updated from API
                regasification_capacity_gwh_d=10.0,  # Default
                metadata={"source": "ALSI"}
            )
            self.assets[terminal_code] = terminal
    
    def discover(self) -> Dict[str, Any]:
        """Discover available data streams from ALSI."""
        
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "granularity": self.granularity,
            "entities": self.entities or "ALL",
            "streams": [
                {
                    "name": "lng_inventory",
                    "variables": [
                        "lng_inventory_gwh",
                        "lng_inventory_mcm",
                        "lng_storage_pct_full",
                        "lng_send_out_gwh",
                        "lng_ship_arrivals",
                    ],
                    "frequency": "daily",
                }
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Fetch LNG data from ALSI API."""
        
        start, end = self._determine_window()
        logger.info(
            "Fetching ALSI LNG data granularity=%s window=%s→%s entities=%s",
            self.granularity,
            start.date(),
            end.date(),
            self.entities or "ALL",
        )
        
        records = list(self._fetch_records(start, end))
        
        # Optionally roll up terminal data to country/EU aggregates
        if self.granularity == "terminal" and self.include_rollups:
            records.extend(self._aggregate_records(records, target="country"))
            records.extend(self._aggregate_records(records, target="eu"))
        
        for record in records:
            yield from self._record_to_events(record)
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw ALSI data to canonical schema."""
        
        ts: datetime = raw["as_of"]
        metric: str = raw["metric"]
        unit: str = raw["unit"]
        entity: str = raw["entity"]
        entity_type: str = raw["entity_type"]
        
        # Get terminal asset if available
        terminal = self.assets.get(entity)
        
        if terminal:
            return self.create_infrastructure_event(
                asset=terminal,
                metric=metric,
                value=raw["value"],
                unit=unit,
                event_time=ts,
                metadata={
                    "entity_type": entity_type,
                    "region": raw.get("region"),
                    "capacity_gwh": raw.get("capacity_gwh"),
                    "ship_arrivals": raw.get("ship_arrivals"),
                }
            )
        else:
            # Fallback for country/EU level data without specific terminal
            instrument_id = self._build_instrument_id(entity, entity_type)
            
            return {
                "event_time_utc": int(ts.timestamp() * 1000),
                "market": "infra",
                "product": metric,
                "instrument_id": instrument_id,
                "location_code": entity,
                "price_type": "observation",
                "value": raw["value"],
                "unit": unit,
                "source": self.source_id,
                "commodity_type": self.commodity_type.value,
                "metadata": json.dumps({
                    "entity_type": entity_type,
                    "region": raw.get("region"),
                    "capacity_gwh": raw.get("capacity_gwh"),
                    "infrastructure_type": InfrastructureType.LNG_TERMINAL.value,
                })
            }
    
    def emit(self, events: Iterable[Dict[str, Any]]) -> int:
        """Emit events to Kafka."""
        return super().emit(events)
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state."""
        super().checkpoint(state)
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _record_to_events(self, record: LNGStorageRecord) -> Iterator[Dict[str, Any]]:
        """Convert LNG storage record to multiple metric events."""
        
        base = {
            "as_of": record.as_of,
            "entity": record.entity,
            "entity_type": record.entity_type,
            "region": record.region,
            "capacity_gwh": record.capacity_gwh,
            "ship_arrivals": record.ship_arrivals,
        }
        
        if record.inventory_gwh is not None:
            yield {**base, "metric": "lng_inventory_gwh", "value": record.inventory_gwh, "unit": "GWh"}
        
        if record.inventory_mcm is not None:
            yield {**base, "metric": "lng_inventory_mcm", "value": record.inventory_mcm, "unit": "mcm"}
        
        if record.fullness_pct is not None:
            yield {**base, "metric": "lng_storage_pct_full", "value": record.fullness_pct, "unit": "pct"}
        
        if record.send_out_gwh is not None:
            yield {**base, "metric": "lng_send_out_gwh", "value": record.send_out_gwh, "unit": "GWh"}
        
        if record.ship_arrivals is not None:
            yield {**base, "metric": "lng_ship_arrivals", "value": float(record.ship_arrivals), "unit": "count"}
    
    def _build_instrument_id(self, entity: str, entity_type: str) -> str:
        """Build instrument ID for LNG infrastructure."""
        
        if entity_type == "terminal":
            return f"ALSI.{entity}.LNG_TERMINAL"
        elif entity_type == "country":
            return f"ALSI.{entity}.LNG_COUNTRY"
        else:
            return "ALSI.EU.LNG_TOTAL"
    
    def _determine_window(self) -> tuple[datetime, datetime]:
        """Determine data fetch window based on checkpoint."""
        
        state = self.checkpoint_state or {}
        last_event_ms = state.get("last_event_time")
        
        if last_event_ms:
            start = datetime.fromtimestamp(last_event_ms / 1000, tz=timezone.utc)
        else:
            start = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        
        end = datetime.now(timezone.utc)
        return start.replace(hour=0, minute=0, second=0, microsecond=0), end
    
    def _fetch_records(self, start: datetime, end: datetime) -> Iterator[LNGStorageRecord]:
        """Fetch LNG records from ALSI API."""
        
        params = self._build_params(start, end)
        endpoint = self._endpoint_for_granularity()
        
        try:
            payload = self._request_json(endpoint, params)
        except RetryError as exc:
            logger.error("ALSI LNG request failed after retries: %s", exc)
            raise
        
        data = payload.get("data") or []
        logger.info("ALSI LNG returned %s rows", len(data))
        
        for row in data:
            record = self._parse_row(row)
            if record:
                yield record
    
    def _build_params(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Build API request parameters."""
        
        params: Dict[str, Any] = {
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
        }
        
        if self.entities:
            key = {
                "terminal": "terminal",
                "country": "country",
                "eu": "country",
            }[self.granularity]
            params[key] = ",".join(self.entities)
        
        return params
    
    def _endpoint_for_granularity(self) -> str:
        """Get API endpoint for granularity level."""
        
        endpoints = {
            "terminal": f"{self.api_base}/terminals",
            "country": f"{self.api_base}/countries", 
            "eu": f"{self.api_base}/eu",
        }
        
        endpoint = endpoints.get(self.granularity)
        if not endpoint:
            raise ValueError(f"Unsupported granularity {self.granularity}")
        
        return endpoint
    
    def _parse_row(self, row: Dict[str, Any]) -> Optional[LNGStorageRecord]:
        """Parse ALSI API row into LNG storage record."""
        
        try:
            gas_day = row.get("gasDayStart")
            if not gas_day:
                return None
            
            as_of = datetime.fromisoformat(f"{gas_day}T00:00:00+00:00")
            
            entity_code = row.get("code") or row.get("key") or row.get("name")
            if not entity_code:
                return None
            
            entity_type = self._infer_entity_type(row)
            country = row.get("country") or row.get("name")
            
            # Parse values with unit conversion
            inventory_gwh = self._safe_float(row.get("lngInventory"))
            inventory_mcm = self._safe_float(row.get("lngInventoryMcm"))
            fullness = self._safe_float(row.get("full"))
            send_out = self._safe_float(row.get("sendOut"))
            arrivals = self._safe_int(row.get("numberOfShipArrivals"))
            capacity_gwh = self._safe_float(row.get("storageTankCapacity"))
            capacity_mcm = self._safe_float(row.get("storageTankCapacityMcm"))
            
            # Update terminal capacity if we have better data
            if entity_type == "terminal" and entity_code in self.assets:
                terminal = self.assets[entity_code]
                if capacity_gwh:
                    terminal.storage_capacity_gwh = capacity_gwh
            
            return LNGStorageRecord(
                as_of=as_of,
                entity=entity_code,
                entity_type=entity_type,
                inventory_gwh=inventory_gwh,
                inventory_mcm=inventory_mcm,
                fullness_pct=fullness,
                send_out_gwh=send_out,
                ship_arrivals=arrivals,
                capacity_gwh=capacity_gwh,
                capacity_mcm=capacity_mcm,
                region=country,
                raw=row,
            )
            
        except Exception as exc:
            logger.warning("Failed to parse ALSI row %s: %s", row, exc)
            return None
    
    def _aggregate_records(self, records: List[LNGStorageRecord], target: str) -> List[LNGStorageRecord]:
        """Aggregate terminal records to country or EU level."""
        
        if not records:
            return []
        
        aggregates: Dict[tuple[str, str], Dict[str, Any]] = {}
        
        for record in records:
            if not record.as_of:
                continue
            
            if target == "country" and record.region:
                key_entity = record.region
            elif target == "country":
                continue
            else:
                key_entity = "EU"
            
            key = (record.as_of.isoformat(), key_entity)
            agg = aggregates.setdefault(
                key,
                {
                    "inventory_gwh": 0.0,
                    "inventory_mcm": 0.0,
                    "capacity_gwh": 0.0,
                    "capacity_mcm": 0.0,
                    "send_out_gwh": 0.0,
                    "ship_arrivals": 0,
                    "fullness_sum": 0.0,
                    "fullness_count": 0,
                },
            )
            
            if record.inventory_gwh is not None:
                agg["inventory_gwh"] += record.inventory_gwh
            if record.inventory_mcm is not None:
                agg["inventory_mcm"] += record.inventory_mcm
            if record.capacity_gwh is not None:
                agg["capacity_gwh"] += record.capacity_gwh
            if record.capacity_mcm is not None:
                agg["capacity_mcm"] += record.capacity_mcm
            if record.send_out_gwh is not None:
                agg["send_out_gwh"] += record.send_out_gwh
            if record.ship_arrivals is not None:
                agg["ship_arrivals"] += record.ship_arrivals
            if record.fullness_pct is not None:
                agg["fullness_sum"] += record.fullness_pct
                agg["fullness_count"] += 1
        
        aggregated_records: List[LNGStorageRecord] = []
        
        for (iso_ts, entity), metrics in aggregates.items():
            as_of = datetime.fromisoformat(iso_ts)
            fullness_avg = (
                metrics["fullness_sum"] / metrics["fullness_count"]
                if metrics["fullness_count"]
                else None
            )
            entity_type = "country" if target == "country" else "eu"
            
            aggregated_records.append(
                LNGStorageRecord(
                    as_of=as_of,
                    entity=entity,
                    entity_type=entity_type,
                    inventory_gwh=metrics["inventory_gwh"],
                    inventory_mcm=metrics["inventory_mcm"],
                    fullness_pct=fullness_avg,
                    send_out_gwh=metrics["send_out_gwh"],
                    ship_arrivals=metrics["ship_arrivals"],
                    capacity_gwh=metrics["capacity_gwh"],
                    capacity_mcm=metrics["capacity_mcm"],
                    region=None if target == "eu" else entity,
                    raw={"aggregated": True, "target": target},
                )
            )
        
        return aggregated_records
    
    def _infer_entity_type(self, row: Dict[str, Any]) -> str:
        """Infer entity type from API row."""
        
        if self.granularity in {"terminal", "country", "eu"}:
            return self.granularity
        if row.get("terminalName"):
            return "terminal"
        if row.get("country"):
            return "country"
        return "eu"
    
    def _safe_float(self, value: Optional[Any]) -> Optional[float]:
        """Safely convert value to float."""
        
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    
    def _safe_int(self, value: Optional[Any]) -> Optional[int]:
        """Safely convert value to int."""
        
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    
    def _validate_config(self) -> None:
        """Validate connector configuration."""
        
        if not self.api_key:
            raise ValueError("ALSI API key must be provided")
        if self.granularity not in {"terminal", "country", "eu"}:
            raise ValueError("granularity must be terminal|country|eu")
    
    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(
            (requests.HTTPError, requests.ConnectionError, requests.Timeout)
        ),
        reraise=True,
    )
    def _request_json(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with retries."""
        
        headers = {
            "x-key": self.api_key,
            "Accept": "application/json",
        }
        
        resp = self.session.get(endpoint, params=params, headers=headers, timeout=30)
        
        if resp.status_code == 401:
            raise requests.HTTPError("Unauthorized access to ALSI API")
        
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "15"))
            logger.warning("ALSI rate limit hit, sleeping %s seconds", retry_after)
            time.sleep(retry_after)
        
        resp.raise_for_status()
        return resp.json()
