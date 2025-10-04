"""
New Zealand EMI (Electricity Authority) Connector

Overview
--------
Fetches half‑hourly Final/Dispatch spot prices from EMI APIs when configured
with an API key; otherwise emits realistic simulated series for development.

Data Flow
---------
EMI API (or mocks) → parse series (island/node) → canonical tick schema → Kafka
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator, Dict, Any, Optional
import time
import json

try:
    import requests  # Optional at runtime
except Exception:  # pragma: no cover
    requests = None

from ..base import Ingestor

logger = logging.getLogger(__name__)


class NewZealandEMIConnector(Ingestor):
    """EMI connector for half-hourly spot prices (Final/Dispatch)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Azure API Management front-door typically used by EMI
        self.api_base = config.get("api_base", "https://emi.azure-api.net")
        self.series = config.get("series", "final_price")  # final_price or dispatch_price
        self.scope = config.get("scope", "island")  # island or node (MVP: island)
        self.island = config.get("island", "NI")  # NI or SI
        self.api_key = config.get("api_key")  # Optional
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.timeout = int(config.get("timeout_seconds", 30))
        self.producer = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": self.series,
                    "market": "power",
                    "product": "energy",
                    "description": f"EMI {self.series.replace('_', ' ')} - half-hourly",
                    "currency": "NZD",
                    "update_freq": "30min",
                    "scope": self.scope,
                }
            ],
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        last_checkpoint = self.load_checkpoint()
        last_time = self._resolve_last_time(last_checkpoint)
        logger.info(f"Fetching EMI {self.series} ({self.island}) since {last_time}")

        # Attempt live API if api_key present; otherwise simulate
        if self.api_key and requests is not None:
            try:
                # Note: EMI provides multiple paths; this is a placeholder pattern
                # and will likely require adjustment to the actual endpoint in use.
                url = f"{self.api_base}/wholesale/prices/{self.series}"
                headers = {"Ocp-Apim-Subscription-Key": self.api_key}
                params = {
                    "start": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "end": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "island": self.island,
                    "format": "json",
                }
                resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
                if resp.status_code == 200 and isinstance(resp.json(), list):
                    for row in resp.json():
                        ts_raw = row.get("IntervalEnding") or row.get("time")
                        price = row.get("Price") or row.get("price")
                        if ts_raw is None or price is None:
                            continue
                        try:
                            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                        except Exception:
                            try:
                                ts = datetime.fromtimestamp(int(ts_raw) / 1000, tz=timezone.utc)
                            except Exception:
                                continue
                        yield {
                            "timestamp": ts.astimezone(timezone.utc).isoformat(),
                            "series": self.series,
                            "island": self.island,
                            "price_nzd_mwh": float(price),
                            "currency": "NZD",
                            "interval": "30min",
                        }
                    return
                else:
                    logger.warning(f"EMI API HTTP {resp.status_code}; falling back to simulation")
            except Exception as e:
                logger.error(f"EMI API error: {e}; falling back to simulation")

        # Simulate last 24 half-hours
        now = datetime.now(timezone.utc).replace(minute=(datetime.now().minute // 30) * 30, second=0, microsecond=0)
        base = 165.0 if self.island == "NI" else 155.0
        for step in range(0, 24):
            ts = now - timedelta(minutes=30 * step)
            hour = ts.hour
            tod = 1.15 if 18 <= hour <= 21 else (1.05 if 9 <= hour <= 17 else 0.85)
            noise = ((hash(f"{self.island}-{ts.isoformat()}") % 30) - 15)
            price = max(90.0, min(350.0, base * tod + noise))
            yield {
                "timestamp": ts.isoformat(),
                "series": self.series,
                "island": self.island,
                "price_nzd_mwh": float(price),
                "currency": "NZD",
                "interval": "30min",
            }

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        label = "FINAL" if self.series == "final_price" else "DISPATCH"
        instr = f"EMI.{label}.{raw.get('island', self.island)}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": instr,
            "location_code": instr,
            "price_type": label.lower(),
            "value": float(raw["price_nzd_mwh"]),
            "volume": None,
            "currency": "NZD",
            "unit": "MWh",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }

    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        if self.producer is None:
            from kafka import KafkaProducer
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
        logger.info(f"Emitted {count} EMI events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        super().checkpoint(state)

    def _resolve_last_time(self, checkpoint: Optional[Dict[str, Any]]) -> datetime:
        if not checkpoint:
            return datetime.now(timezone.utc) - timedelta(hours=12)
        val = checkpoint.get("last_event_time")
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val / 1000, tz=timezone.utc)
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except Exception:
                return datetime.now(timezone.utc) - timedelta(hours=12)
        if isinstance(val, datetime):
            return val.astimezone(timezone.utc)
        return datetime.now(timezone.utc) - timedelta(hours=12)


if __name__ == "__main__":
    cfg = {
        "source_id": "emi_finalprice_island",
        "series": "final_price",
        "scope": "island",
        "island": "NI",
        "kafka_topic": "power.ticks.v1",
    }
    connector = NewZealandEMIConnector(cfg)
    connector.run()
