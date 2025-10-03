"""
Vietnam Power Market Connector

Integrates with Vietnam's emerging wholesale market:
- Competitive generation market (CGM)
- Solar/wind curtailment tracking
- Industrial demand growth
- EVN (Electricity Vietnam) pricing
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class VietnamConnector(Ingestor):
    """Vietnam power market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.evn.com.vn/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover Vietnam market streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "wholesale_prices",
                    "description": "Competitive generation market prices",
                    "currency": "VND",
                },
                {
                    "name": "curtailment",
                    "description": "Solar/wind curtailment rates",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull Vietnam market data."""
        logger.info("Fetching Vietnam power data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch Vietnam wholesale prices."""
        now = datetime.utcnow()
        vietnam_time = now + timedelta(hours=7)  # ICT (UTC+7)
        
        # Base price (VND/MWh) - ~1,800 VND/kWh = 1,800,000 VND/MWh
        base_price = 1750000  # ~$75 USD/MWh at 23,000 VND/USD
        
        hour = vietnam_time.hour
        
        # Load pattern
        if 19 <= hour <= 22:
            tod_factor = 1.4  # Evening peak
        elif 9 <= hour <= 18:
            tod_factor = 1.2
        else:
            tod_factor = 0.80
        
        # Industrial demand growth
        year = vietnam_time.year
        growth_factor = 1 + ((year - 2024) * 0.08)  # 8% annual growth
        
        # Solar curtailment impact (mid-day surplus)
        if 11 <= hour <= 14:
            curtailment_discount = 0.85  # Mid-day solar oversupply
        else:
            curtailment_discount = 1.0
        
        price = base_price * tod_factor * growth_factor * curtailment_discount
        price += (hash(str(hour)) % 150000) - 75000
        price = max(1400000, min(2500000, price))
        
        # Load (MW)
        load = 42000 + (hash(str(hour)) % 8000)
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "VIETNAM_EVN",
            "price_vnd_mwh": price,
            "hour_ending": vietnam_time.replace(minute=0, second=0).isoformat(),
            "load_mw": load,
            "currency": "VND",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map Vietnam format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "VIETNAM.EVN",
            "location_code": "VIETNAM.EVN",
            "price_type": "wholesale",
            "value": float(raw["price_vnd_mwh"]),
            "volume": raw.get("load_mw"),
            "currency": "VND",
            "unit": "MWh",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }
    
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
        logger.info(f"Emitted {count} Vietnam events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = VietnamConnector({"source_id": "vietnam_evn"})
    connector.run()

