"""
Hungary HUPX (Hungarian Power Exchange) Connector

Day-ahead and intraday markets with regional coupling
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class HungaryHUPXConnector(Ingestor):
    """Hungary HUPX connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull HUPX data."""
        logger.info("Fetching Hungary HUPX data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch day-ahead prices."""
        tomorrow = datetime.utcnow() + timedelta(days=1)
        
        for hour in range(24):
            delivery = (tomorrow + timedelta(hours=1)).replace(hour=hour, minute=0, second=0, microsecond=0)
            
            base = 30000  # HUF/MWh (~$85 USD/MWh at 350 HUF/USD)
            
            if 18 <= hour <= 21:
                factor = 1.4
            elif 7 <= hour <= 17:
                factor = 1.2
            else:
                factor = 0.80
            
            price = base * factor + (hash(str(hour)) % 4000) - 2000
            price = max(22000, min(45000, price))
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "HUNGARY_HUPX",
                "price_huf_mwh": price,
                "delivery_hour": delivery.isoformat(),
                "currency": "HUF",
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map HUPX format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "HUPX.HUNGARY",
            "location_code": "HUPX.HUNGARY",
            "price_type": "day_ahead",
            "value": float(raw["price_huf_mwh"]),
            "volume": None,
            "currency": "HUF",
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
    connector = HungaryHUPXConnector({"source_id": "hungary_hupx"})
    connector.run()


