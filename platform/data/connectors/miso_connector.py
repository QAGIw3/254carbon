"""
MISO LMP Connector

Overview
--------
Pulls Real‑Time (5‑minute nodal) and Day‑Ahead (ex‑ante hub) LMPs from the
public MISO Real‑Time Web Displays Data Broker (MISORTWD). Normalizes New York
local timestamps to UTC and emits canonical events to Kafka.

Docs and endpoints (public)
- Index of JSON/CSV/XML endpoints:
  https://api.misoenergy.org/MISORTWDDataBroker/
- Real‑time consolidated LMP table (5‑minute RT):
  https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx?messageType=getlmpconsolidatedtable&returnType=json
- Ex‑ante LMP (hub‑level DA):
  https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx?messageType=getexantelmp&returnType=json

Notes
-----
- The consolidated table response provides a ref date and `HourAndMin` field
  in America/New_York; we convert to UTC.
- MISORTWD fields `MCC`/`MLC` map to congestion/loss components.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator, Dict, Any, Optional
import time

import requests
from kafka import KafkaProducer
import json
from zoneinfo import ZoneInfo

from .base import Ingestor

logger = logging.getLogger(__name__)


class MISOConnector(Ingestor):
    """
    MISO RT 5-min nodal and DA (ex-ante) hub LMP connector.

    Responsibilities
    - Discover streams exposed by MISORTWD Data Broker (RT nodal, DA ex-ante hubs)
    - Pull JSON payloads from official endpoints with retry/backoff
    - Normalize timestamps (EST/EDT → UTC) and map to canonical schema
    - Promote component fields (MCC/MLC) into congestion/loss component keys
    - Emit canonical events to Kafka with basic validation from base Ingestor

    Configuration
    - api_base: Base URL for MISORTWD Data Broker
    - market_type: "RT" for real-time 5-min nodal; "DA" for ex-ante hubs
    - kafka_topic, kafka_bootstrap: Emission settings
    - timeout_seconds, max_retries, retry_backoff_base: HTTP behavior
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # API and emission configuration
        self.api_base = config.get("api_base", "https://api.misoenergy.org/MISORTWDDataBroker")
        self.market_type = config.get("market_type", "RT")  # RT or DA (DA uses ExAnte hubs)
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
        # networking
        self.timeout_seconds: int = int(config.get("timeout_seconds", 30))
        self.max_retries: int = int(config.get("max_retries", 3))
        self.retry_backoff_base: float = float(config.get("retry_backoff_base", 1.0))
    
    def discover(self) -> Dict[str, Any]:
        """
        Discover MISO streams available via MISORTWD Data Broker.

        Returns a static description that documents the source endpoints
        and cadence. The actual pull method switches based on market_type.
        """
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "rt_fivemin_lmp_nodal",
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "5min",
                    "approx_nodes": "~3000",
                    "endpoint": "getlmpconsolidatedtable",
                    "format": "json",
                },
                {
                    "name": "da_exante_lmp_hub",
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "hourly",
                    "approx_hubs": 9,
                    "endpoint": "getexantelmp",
                    "format": "json",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull LMP data from MISO MISORTWD.
        
        For RT: polls every 5 minutes
        For DA: polls hourly for next day
        """
        last_checkpoint = self.load_checkpoint()
        last_time = self._resolve_last_time(last_checkpoint)
        
        logger.info(f"Fetching MISO {self.market_type} LMP from MISORTWD Data Broker")

        if self.market_type.upper() == "RT":
            yield from self._fetch_realtime_fivemin_lmp()
        else:
            yield from self._fetch_exante_hub_lmp()

    def _fetch_realtime_fivemin_lmp(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch current 5-minute nodal LMPs from the consolidated table.

        Endpoint returns an object with metadata, an HourAndMin field, and
        a list of PricingNode entries containing LMP, MLC (loss), MCC (congestion).
        We parse the reference time in America/New_York and convert to UTC.
        """
        url = (
            f"{self.api_base}/DataBrokerServices.asmx"
            f"?messageType=getlmpconsolidatedtable&returnType=json"
        )
        attempt = 0
        while True:
            attempt += 1
            resp = requests.get(url, timeout=self.timeout_seconds)
            if resp.status_code == 200:
                break
            if attempt > self.max_retries:
                raise RuntimeError(f"MISO DataBroker HTTP {resp.status_code}")
            time.sleep(self.retry_backoff_base * (2 ** (attempt - 1)))

        payload = resp.json()
        lmpdata = payload.get("LMPData", {})
        # Example RefId: "03-Oct-2025 - Interval 19:45 EST"
        ref_id: str = lmpdata.get("RefId", "")
        date_part = ref_id.split(" - ")[0] if " - " in ref_id else None
        hm = ((lmpdata.get("FiveMinLMP") or {}).get("HourAndMin"))
        try:
            if date_part and hm:
                dt_local = datetime.strptime(f"{date_part} {hm}", "%d-%b-%Y %H:%M").replace(
                    tzinfo=ZoneInfo("America/New_York")
                )
                ts = dt_local.astimezone(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)

        nodes = ((lmpdata.get("FiveMinLMP") or {}).get("PricingNode")) or []
        for row in nodes:
            try:
                node = row.get("name") or row.get("PricingNode")
                if not node:
                    continue
                lmp = float(row.get("LMP")) if row.get("LMP") not in (None, "") else None
                mlc = row.get("MLC")
                mcc = row.get("MCC")
                yield {
                    "timestamp": ts.isoformat(),
                    "node_id": f"MISO.{node}",
                    "lmp": lmp,
                    # map to canonical component field names later
                    "mlc": float(mlc) if mlc not in (None, "") else None,
                    "mcc": float(mcc) if mcc not in (None, "") else None,
                    "market": "RT",
                    "interval": "5min",
                }
            except Exception as ex:
                logger.debug(f"Skipping malformed RT node row: {ex}")

    def _fetch_exante_hub_lmp(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch ex-ante (day-ahead-like) hub LMPs.

        The ex-ante series is hub-level and provides a consolidated view
        with LMP and separate loss/congestion components.
        """
        url = (
            f"{self.api_base}/DataBrokerServices.asmx"
            f"?messageType=getexantelmp&returnType=json"
        )
        attempt = 0
        while True:
            attempt += 1
            resp = requests.get(url, timeout=self.timeout_seconds)
            if resp.status_code == 200:
                break
            if attempt > self.max_retries:
                raise RuntimeError(f"MISO DataBroker HTTP {resp.status_code}")
            time.sleep(self.retry_backoff_base * (2 ** (attempt - 1)))

        payload = resp.json()
        lmpdata = payload.get("LMPData", {})
        ref_id: str = lmpdata.get("RefId", "")
        date_part = ref_id.split(" - ")[0] if " - " in ref_id else None
        hm = ((lmpdata.get("ExAnteLMP") or {}).get("HourAndMin"))
        try:
            if date_part and hm:
                dt_local = datetime.strptime(f"{date_part} {hm}", "%d-%b-%Y %H:%M").replace(
                    tzinfo=ZoneInfo("America/New_York")
                )
                ts = dt_local.astimezone(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)

        hubs = ((lmpdata.get("ExAnteLMP") or {}).get("Hub")) or []
        for row in hubs:
            try:
                name = row.get("name")
                if not name:
                    continue
                lmp = float(row.get("LMP")) if row.get("LMP") not in (None, "") else None
                loss = row.get("loss")
                cong = row.get("congestion")
                yield {
                    "timestamp": ts.isoformat(),
                    "node_id": f"MISO.{name}",
                    "lmp": lmp,
                    "mlc": float(loss) if loss not in (None, "") else None,
                    "mcc": float(cong) if cong not in (None, "") else None,
                    "market": "DA",
                    "interval": "hourly",
                }
            except Exception as ex:
                logger.debug(f"Skipping malformed ExAnte hub row: {ex}")
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map MISO raw record to canonical schema.

        - event_time_utc: epoch ms
        - instrument_id: namespaced node/hub id (we preserve the MISO.* prefix
          upstream when building the raw payload to avoid later ambiguity)
        - price_type: settle for DA (ex-ante), trade for RT
        - value: LMP; components added when provided
        """
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        payload = {
            "event_time_utc": int(timestamp.timestamp() * 1000),  # milliseconds
            "market": "power",
            "product": "lmp",
            "instrument_id": raw["node_id"],
            "location_code": raw["node_id"],
            "price_type": "settle" if raw["market"] == "DA" else "trade",
            "value": float(raw["lmp"]),
            "volume": None,
            "currency": "USD",
            "unit": "MWh",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }
        # Promote components if present in raw
        if raw.get("energy_component") is not None:
            payload["energy_component"] = float(raw["energy_component"])  # type: ignore[arg-type]
        # MISORTWD fields map to MCC/MLC naming
        if raw.get("mcc") is not None:
            payload["congestion_component"] = float(raw["mcc"])  # type: ignore[arg-type]
        if raw.get("mlc") is not None:
            payload["loss_component"] = float(raw["mlc"])  # type: ignore[arg-type]
        return payload

    def _resolve_last_time(self, checkpoint: Optional[Dict[str, Any]]) -> datetime:
        """Resolve the reference timestamp for incremental pulling."""
        if not checkpoint:
            return datetime.now(timezone.utc) - timedelta(hours=1)

        last_event_time = checkpoint.get("last_event_time")

        if last_event_time is None:
            return datetime.now(timezone.utc) - timedelta(hours=1)

        if isinstance(last_event_time, (int, float)):
            return datetime.fromtimestamp(last_event_time / 1000, tz=timezone.utc)

        if isinstance(last_event_time, str):
            try:
                return datetime.fromisoformat(last_event_time)
            except ValueError:
                logger.warning("Invalid last_event_time in checkpoint, defaulting to 1 hour lookback")
                return datetime.now(timezone.utc) - timedelta(hours=1)

        if isinstance(last_event_time, datetime):
            return last_event_time.astimezone(timezone.utc)

        logger.warning("Unsupported last_event_time type in checkpoint, defaulting to 1 hour lookback")
        return datetime.now(timezone.utc) - timedelta(hours=1)
    
    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """Emit events to Kafka (lazy producer + flush for delivery)."""
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
        logger.info(f"Emitted {count} events to {self.kafka_topic}")
        return count
    


if __name__ == "__main__":
    # Test connector
    config = {
        "source_id": "miso_rt_lmp",
        "market_type": "RT",
        "kafka_topic": "power.ticks.v1",
    }
    
    connector = MISOConnector(config)
    connector.run()
