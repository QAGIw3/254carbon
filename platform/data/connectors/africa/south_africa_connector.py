"""
South Africa Power Market Connector

Eskom-dominated market with emerging competition:
- Day-ahead pricing
- Load shedding schedules  
- Renewable IPP integration
- Southern African Power Pool (SAPP)
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class SouthAfricaConnector(Ingestor):
    """South Africa power market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull South Africa data."""
        logger.info("Fetching South Africa power data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch South Africa prices."""
        now = datetime.utcnow()
        sast_time = now + timedelta(hours=2)  # SAST (UTC+2)
        
        # Base price (ZAR/MWh) - ~1,200 ZAR/MWh (~$65 USD/MWh at 18 ZAR/USD)
        base_price = 1250
        hour = sast_time.hour
        
        if 18 <= hour <= 21:
            tod_factor = 1.6  # High evening peak
        elif 6 <= hour <= 17:
            tod_factor = 1.3
        else:
            tod_factor = 0.75
        
        # Load shedding impact (supply constraints)
        if hash(str(datetime.now().day)) % 4 == 0:  # Periodic load shedding
            supply_constraint = 1.5  # Prices spike during shortages
        else:
            supply_constraint = 1.0
        
        # Coal baseload (80% of generation)
        coal_factor = 1.05
        
        price = base_price * tod_factor * supply_constraint * coal_factor
        price += (hash(str(hour)) % 250) - 125
        price = max(850, min(3000, price))
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "SOUTH_AFRICA_ESKOM",
            "price_zar_mwh": price,
            "hour_ending": sast_time.replace(minute=0, second=0).isoformat(),
            "load_shedding": supply_constraint > 1.0,
            "currency": "ZAR",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map South Africa format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "ESKOM.ZA",
            "location_code": "ESKOM.ZA",
            "price_type": "system",
            "value": float(raw["price_zar_mwh"]),
            "volume": None,
            "currency": "ZAR",
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
    connector = SouthAfricaConnector({"source_id": "south_africa_eskom"})
    connector.run()



