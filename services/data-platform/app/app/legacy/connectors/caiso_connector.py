"""
CAISO Nodal LMP Connector

Overview
--------
Pulls Real‑Time (RTM) and Day‑Ahead (DAM) LMP data from CAISO OASIS using the
SingleZip endpoint (CSV‑in‑ZIP) for reliable parsing. Enforces entitlement
restrictions (hub‑only access) for pilot customers. Emits canonical events to
Kafka for downstream persistence and analytics.

Data Flow
---------
CAISO OASIS SingleZip → CSV parser → canonical tick schema → Kafka topic

Configuration
-------------
- ``api_base``: OASIS base URL (SingleZip)
- ``market_type``: ``RTM`` or ``DAM``
- ``hub_only`` / ``allowed_hubs`` / ``allowed_nodes``: entitlement controls
- ``timeout_seconds`` / ``max_retries`` / ``retry_backoff_base``: network tuning
- ``override_start`` / ``override_end``: optional backfill window

Operational Notes
-----------------
- Timestamps are normalized from OASIS fields to UTC.
- CSV shape varies slightly by report; column aliases are handled defensively.
- Retries use exponential backoff with a base factor; client errors (4xx)
  terminate early to avoid waste.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator, Dict, Any, Optional, List
import time
import io
import zipfile
import csv

import requests
from kafka import KafkaProducer
import json

from .base import Ingestor

logger = logging.getLogger(__name__)


class CAISOConnector(Ingestor):
    """
    CAISO nodal DA/RT LMP connector with entitlement support.

    Responsibilities
    - Discover and pull RTM/DAM LMPs via OASIS SingleZip (CSV-in-ZIP)
    - Normalize timestamps and fields from CSV rows into canonical schema
    - Enforce entitlement restrictions (pilot hub-only access)
    - Emit to Kafka with validation and sequence ordering

    Configuration highlights
    - api_base: OASIS SingleZip base URL
    - market_type: RTM (real-time) or DAM (day-ahead)
    - hub_only / allowed_hubs / allowed_nodes: entitlement/allowlist controls
    - timeout_seconds / max_retries / retry_backoff_base: network behavior
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Network and API configuration
        self.api_base = config.get("api_base", "https://oasis.caiso.com/oasisapi/SingleZip")
        self.market_type = config.get("market_type", "RTM")  # RTM (Real-Time) or DAM (Day-Ahead)
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.entitlements_enabled = config.get("entitlements_enabled", True)
        self.producer = None
        # Networking and retry settings
        self.timeout_seconds: int = int(config.get("timeout_seconds", 30))
        self.max_retries: int = int(config.get("max_retries", 3))
        self.retry_backoff_base: float = float(config.get("retry_backoff_base", 1.0))
        self.dev_mode: bool = bool(config.get("dev_mode", False))
        # Optional backfill window overrides (datetime or OASIS-formatted string)
        self.override_start = config.get("override_start")
        self.override_end = config.get("override_end")
        
        # CAISO-specific configuration
        self.price_nodes = config.get("price_nodes", "ALL")  # ALL or specific nodes
        self.hub_only = config.get("hub_only", True)  # For pilot: only hub data
        self.allowed_hubs: List[str] = [
            "TH_SP15_GEN-APND",
            "TH_NP15_GEN-APND",
            "TH_ZP26_GEN-APND",
        ]
        # Optional controlled allowlist for nodal expansion
        self.allowed_nodes: Optional[List[str]] = config.get("allowed_nodes")
    
    def discover(self) -> Dict[str, Any]:
        """
        Discover CAISO pricing points and hubs.

        Returns a metadata snapshot indicating the scope of data we
        will request from OASIS and how entitlements constrain it.
        """
        # Major CAISO trading hubs
        hubs = [
            "TH_SP15_GEN-APND",  # SP15 Trading Hub
            "TH_NP15_GEN-APND",  # NP15 Trading Hub
            "TH_ZP26_GEN-APND",  # ZP26 Trading Hub
        ]
        
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "nodal_lmp",
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "5min" if self.market_type == "RTM" else "1hour",
                    "hubs": len(hubs),
                    "nodes": "~6000" if not self.hub_only else f"{len(hubs)} hubs",
                    "entitlements": "hub+downloads only, API disabled" if self.entitlements_enabled else "full",
                }
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull LMP data from CAISO OASIS.

        Behavior
        - RTM: 5-minute intervals
        - DAM: hourly/day-ahead intervals
        - Entitlements: hub-only restriction for pilot customers unless disabled
        """
        last_checkpoint = self.load_checkpoint()
        last_time = self._resolve_last_time(last_checkpoint)
        
        logger.info(f"Fetching CAISO {self.market_type} LMP since {last_time}")
        
        # Define trading hubs for pilot access
        if self.hub_only:
            nodes = [
                "TH_SP15_GEN-APND",  # SP15 Trading Hub
                "TH_NP15_GEN-APND",  # NP15 Trading Hub
                "TH_ZP26_GEN-APND",  # ZP26 Trading Hub
            ]
        else:
            # Full nodal access (not available in pilot)
            nodes = [f"CAISO.NODE.{i:04d}" for i in range(1, 101)]  # Sample nodes
        
        # Query CAISO OASIS API for each trading hub
        for node in nodes:
            # Enforce hub-only entitlement when enabled
            if self.entitlements_enabled and self.hub_only and node not in self.allowed_hubs:
                logger.debug(f"Skipping unauthorized node under hub-only restriction: {node}")
                continue
            # If an explicit allowlist is provided, enforce it
            if self.allowed_nodes is not None and node not in self.allowed_nodes:
                logger.debug(f"Skipping node not in allowed_nodes: {node}")
                continue

            try:
                yield from self._fetch_oasis_csv_for_node(node)
            except Exception as e:
                logger.error(f"Error querying CAISO OASIS for {node}: {e}")
                if self.dev_mode:
                    # Minimal dev-mode mock with components
                    current_time = datetime.now(timezone.utc)
                    base_price = 40.00
                    energy = base_price * 0.9
                    cong = base_price * 0.07
                    loss = base_price * 0.03
                    yield {
                        "timestamp": current_time.isoformat(),
                        "node_id": node,
                        "lmp": round(base_price, 2),
                        "energy_component": round(energy, 2),
                        "congestion_component": round(cong, 2),
                        "loss_component": round(loss, 2),
                        "market": self.market_type,
                        "interval": "5min" if self.market_type == "RTM" else "hourly",
                    }
                else:
                    continue

    def _fetch_oasis_csv_for_node(self, node: str) -> Iterator[Dict[str, Any]]:
        """
        Fetch CAISO OASIS SingleZip CSV for one node and yield raw records.

        Uses resultformat=6 (CSV-in-ZIP) for robust, schema-stable ingestion.
        Retries with exponential backoff and bails early on 4xx auth errors.
        """
        now_utc = datetime.now(timezone.utc)
        headers = {"User-Agent": "254Carbon-Platform/1.0"}

        # Determine query name and default window
        if self.market_type == "RTM":
            queryname = "PRC_RTM_LMP"
            market_run_id = "RTM"
            default_start = (now_utc - timedelta(minutes=15))
            default_end = now_utc
        else:
            queryname = "PRC_LMP"
            market_run_id = "DAM"
            default_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            default_end = now_utc.replace(hour=23, minute=59, second=0, microsecond=0)

        # Apply overrides if provided
        start_dt = default_start
        end_dt = default_end
        if self.override_start is not None and self.override_end is not None:
            try:
                if isinstance(self.override_start, datetime):
                    start_dt = self.override_start
                else:
                    start_dt = datetime.fromisoformat(str(self.override_start).replace('Z', '+00:00'))
                if isinstance(self.override_end, datetime):
                    end_dt = self.override_end
                else:
                    end_dt = datetime.fromisoformat(str(self.override_end).replace('Z', '+00:00'))
            except Exception:
                # Fallback to defaults on parse errors
                start_dt = default_start
                end_dt = default_end

        start = start_dt.strftime("%Y%m%dT%H:%M-0000")
        end = end_dt.strftime("%Y%m%dT%H:%M-0000")

        params = {
            "queryname": queryname,
            "version": "1",
            "market_run_id": market_run_id,
            "node": node,
            "startdatetime": start,
            "enddatetime": end,
            "resultformat": "6",
        }

        attempt = 0
        while True:
            attempt += 1
            resp = requests.get(self.api_base, params=params, headers=headers, timeout=self.timeout_seconds)
            if resp.status_code == 200:
                break
            if attempt > self.max_retries or resp.status_code in (400, 401, 403):
                raise RuntimeError(f"CAISO OASIS HTTP {resp.status_code} for node {node}")
            backoff = self.retry_backoff_base * (2 ** (attempt - 1))
            time.sleep(backoff)

        # Parse CSV inside ZIP (OASIS SingleZip response)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = next((n for n in zf.namelist() if n.lower().endswith('.csv')), None)
            if not csv_name:
                raise ValueError("No CSV found in OASIS ZIP response")
            with zf.open(csv_name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8', newline=''))
                for row in reader:
                    try:
                        ts_raw = row.get('INTERVALENDTIME_GMT') or row.get('INTERVALENDTIME') or row.get('OPR_INTERVAL_END')
                        node_id = row.get('NODE') or row.get('PNODE') or node
                        lmp = row.get('LMP_PRC') or row.get('LMP')
                        energy = row.get('ENERGY_PRC') or row.get('ENERGY') or None
                        cong = row.get('CONG_PRC') or row.get('CONGESTION') or None
                        loss = row.get('LOSS_PRC') or row.get('LOSS') or None
                        if ts_raw is None or lmp is None or node_id is None:
                            continue
                        # Normalize timestamp
                        ts = datetime.fromisoformat(str(ts_raw).replace('Z', '+00:00'))
                        rec = {
                            "timestamp": ts.isoformat(),
                            "node_id": node_id,
                            "lmp": float(lmp),
                            "energy_component": float(energy) if energy not in (None, '') else None,
                            "congestion_component": float(cong) if cong not in (None, '') else None,
                            "loss_component": float(loss) if loss not in (None, '') else None,
                            "market": market_run_id,
                            "interval": "5min" if market_run_id == "RTM" else "hourly",
                        }
                        yield rec
                    except Exception as ex:
                        logger.debug(f"Skipping malformed CAISO row for {node}: {ex}")
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map CAISO format to canonical schema.

        This preserves instrument identity as CAISO.<node>, sets price_type by
        market, and promotes component fields where available.
        """
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Create standardized instrument ID
        instrument_id = f"CAISO.{raw['node_id']}"
        
        payload = {
            "event_time_utc": int(timestamp.timestamp() * 1000),  # milliseconds
            "market": "power",
            "product": "lmp",
            "instrument_id": instrument_id,
            "location_code": raw["node_id"],
            "price_type": "settle" if raw["market"] == "DAM" else "trade",
            "value": float(raw["lmp"]),
            "volume": None,
            "currency": "USD",
            "unit": "MWh",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }
        # Promote components when available
        if raw.get("energy_component") is not None:
            payload["energy_component"] = float(raw["energy_component"])  # type: ignore[arg-type]
        if raw.get("congestion_component") is not None:
            payload["congestion_component"] = float(raw["congestion_component"])  # type: ignore[arg-type]
        if raw.get("loss_component") is not None:
            payload["loss_component"] = float(raw["loss_component"])  # type: ignore[arg-type]
        return payload

    def _resolve_last_time(self, checkpoint: Optional[Dict[str, Any]]) -> datetime:
        """Resolve the most recent processed timestamp for incremental pulls."""
        if not checkpoint:
            return datetime.now(timezone.utc) - timedelta(hours=1)

        last_event_time = checkpoint.get("last_event_time")

        if last_event_time is None:
            return datetime.now(timezone.utc) - timedelta(hours=1)

        if isinstance(last_event_time, (int, float)):
            return datetime.fromtimestamp(last_event_time / 1000, tz=timezone.utc)

        if isinstance(last_event_time, str):
            try:
                dt = datetime.fromisoformat(last_event_time)
                return dt.astimezone(timezone.utc)
            except ValueError:
                logger.warning("Invalid checkpoint last_event_time; defaulting to 1 hour lookback")
                return datetime.now(timezone.utc) - timedelta(hours=1)

        if isinstance(last_event_time, datetime):
            return last_event_time.astimezone(timezone.utc)

        logger.warning("Unsupported last_event_time type in checkpoint; defaulting to 1 hour lookback")
        return datetime.now(timezone.utc) - timedelta(hours=1)
    
    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """Emit events to Kafka.

        Notes
        -----
        - Producer is lazily initialized to avoid repeated connections.
        - Events are filtered by entitlement when enabled.
        - Flush ensures delivery before checkpointing.
        """
        if self.producer is None:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        
        count = 0
        for event in events:
            try:
                # Apply entitlement check
                if self.entitlements_enabled and not self._check_entitlement(event):
                    logger.debug(f"Event filtered by entitlements: {event['instrument_id']}")
                    continue
                
                self.producer.send(self.kafka_topic, value=event)
                count += 1
            except Exception as e:
                logger.error(f"Kafka send error: {e}")
        
        self.producer.flush()
        logger.info(f"Emitted {count} events to {self.kafka_topic}")
        return count
    
    def _check_entitlement(self, event: Dict[str, Any]) -> bool:
        """
        Check if event passes entitlement restrictions.
        
        For CAISO pilot: only hub data allowed.
        """
        if not self.hub_only:
            return True
        
        # Allow only trading hub data
        allowed_hubs = [
            "CAISO.TH_SP15_GEN-APND",
            "CAISO.TH_NP15_GEN-APND",
            "CAISO.TH_ZP26_GEN-APND",
        ]
        
        return event["instrument_id"] in allowed_hubs
    


if __name__ == "__main__":
    # Test connector with pilot configuration
    config = {
        "source_id": "caiso_rtm_lmp",
        "market_type": "RTM",
        "kafka_topic": "power.ticks.v1",
        "hub_only": True,  # Pilot restriction
        "entitlements_enabled": True,
    }
    
    connector = CAISOConnector(config)
    
    # Discovery
    discovery = connector.discover()
    print("Discovery:", json.dumps(discovery, indent=2))
    
    # Run ingestion
    connector.run()
