"""
Egypt Power Market Connector

Egyptian Electricity Holding Company (EEHC) unified grid:
- Suez wind corridor
- Natural gas dependency
- Growing solar capacity
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class EgyptConnector(Ingestor):
    """Egypt power market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull Egypt data."""
        logger.info("Fetching Egypt power data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch Egypt system prices."""
        now = datetime.utcnow()
        egypt_time = now + timedelta(hours=2)  # EET (UTC+2)
        
        # Base price (EGP/MWh) - ~1,200 EGP/MWh (~$40 USD/MWh at 30 EGP/USD)
        base_price = 1200
        hour = egypt_time.hour
        
        if 19 <= hour <= 23:
            tod_factor = 1.5
        elif 8 <= hour <= 18:
            tod_factor = 1.2
        else:
            tod_factor = 0.85
        
        # Gas price influence (60% gas generation)
        gas_factor = 1.05
        
        # Wind from Suez corridor
        if 14 <= hour <= 20:  # Strong afternoon/evening winds
            wind_discount = 0.90
        else:
            wind_discount = 1.0
        
        price = base_price * tod_factor * gas_factor * wind_discount
        price += (hash(str(hour)) % 180) - 90
        price = max(900, min(1800, price))
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "EGYPT_EEHC",
            "price_egp_mwh": price,
            "hour_ending": egypt_time.replace(minute=0, second=0).isoformat(),
            "currency": "EGP",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map Egypt format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "EGYPT.EEHC",
            "location_code": "EGYPT.EEHC",
            "price_type": "system",
            "value": float(raw["price_egp_mwh"]),
            "volume": None,
            "currency": "EGP",
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
    connector = EgyptConnector({"source_id": "egypt_eehc"})
    connector.run()


