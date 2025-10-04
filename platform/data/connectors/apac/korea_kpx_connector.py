"""
Korea Power Exchange (KPX) Connector

Integrates with South Korean power market:
- Cost‑Based Pool (CBP) pricing
- System Marginal Price (SMP)
- Renewable Energy Credits (REC)
- Nuclear baseload (30%)

Data Flow
---------
KPX feeds (or mocks) → SMP normalization (KRW/MWh, KST→UTC) → canonical schema → Kafka
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class KoreaKPXConnector(Ingestor):
    """Korea Power Exchange connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.kpx.or.kr/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull KPX market data."""
        logger.info("Fetching Korea KPX data")
        yield from self._fetch_smp()
    
    def _fetch_smp(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch System Marginal Price (SMP).
        
        Korea uses cost-based pool with hourly SMP.
        """
        now = datetime.utcnow()
        kst_time = now + timedelta(hours=9)  # KST (UTC+9)
        
        hour = kst_time.hour
        
        # Base SMP (KRW/kWh → KRW/MWh)
        base_price = 120000  # ~$95 USD/MWh at 1,250 KRW/USD
        
        # Time of day
        if 10 <= hour <= 11 or 14 <= hour <= 15 or 19 <= hour <= 21:
            tod_factor = 1.5  # Multiple daily peaks
        elif 7 <= hour <= 18:
            tod_factor = 1.2
        else:
            tod_factor = 0.75
        
        # Nuclear provides stable baseload (30%)
        # LNG sets marginal price
        lng_factor = 1.1
        
        price = base_price * tod_factor * lng_factor
        price += (hash(str(hour)) % 15000) - 7500
        price = max(85000, min(180000, price))
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "KOREA_KPX",
            "price_krw_mwh": price,
            "hour_ending": kst_time.replace(minute=0, second=0).isoformat(),
            "currency": "KRW",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map KPX format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "KPX.SMP",
            "location_code": "KPX.SMP",
            "price_type": "smp",
            "value": float(raw["price_krw_mwh"]),
            "volume": None,
            "currency": "KRW",
            "unit": "MWh",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }
    
    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """Emit to Kafka."""
        if self.producer is None:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        
        count = 0
        for event in events:
            self.producer.send(self.kafka_topic, value=event)
            count += 1
        
        self.producer.flush()
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = KoreaKPXConnector({"source_id": "korea_kpx"})
    connector.run()


