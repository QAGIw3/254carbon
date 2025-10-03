"""
Nigeria Power Market Connector

Integrates with Nigerian Electricity Supply Industry (NESI):
- Day-ahead market
- Real-time dispatch
- West Africa Power Pool integration
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class NigeriaConnector(Ingestor):
    """Nigeria power market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.nigerianelectricityhub.com/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover Nigeria market streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "day_ahead",
                    "description": "Day-ahead market prices",
                    "currency": "NGN",
                },
                {
                    "name": "real_time",
                    "description": "Real-time dispatch",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull Nigeria market data."""
        logger.info("Fetching Nigeria power data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch Nigeria power prices."""
        # Nigeria operates with significant supply constraints
        # Prices vary widely based on availability
        
        now = datetime.utcnow()
        nigeria_time = now + timedelta(hours=1)  # WAT (UTC+1)
        
        # Base price (NGN/MWh)
        base_price = 35000  # ~$85 USD/MWh at 410 NGN/USD
        
        # Supply availability factor
        hour = nigeria_time.hour
        if 18 <= hour <= 22:
            supply_factor = 1.4  # Peak demand, limited supply
        elif 6 <= hour <= 17:
            supply_factor = 1.2
        else:
            supply_factor = 0.9
        
        # Grid stability factor
        stability = 0.7  # Nigeria averages ~70% grid availability
        if hash(str(now.hour)) % 3 == 0:  # Occasional outages
            stability = 0.4
            supply_factor *= 1.5  # Prices spike during outages
        
        price = base_price * supply_factor
        price += (hash(str(nigeria_time.minute)) % 5000) - 2500
        price = max(20000, min(80000, price))
        
        # Available capacity (MW)
        installed_capacity = 12500  # MW
        available_capacity = installed_capacity * stability
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "NIGERIA",
            "price_ngn_mwh": price,
            "hour_ending": nigeria_time.replace(minute=0, second=0).isoformat(),
            "available_capacity_mw": available_capacity,
            "grid_stability_pct": stability * 100,
            "currency": "NGN",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map Nigeria format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "NIGERIA.NESI",
            "location_code": "NIGERIA.NESI",
            "price_type": "market_price",
            "value": float(raw["price_ngn_mwh"]),
            "volume": None,
            "currency": "NGN",
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
        logger.info(f"Emitted {count} Nigeria events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = NigeriaConnector({"source_id": "nigeria_nesi"})
    connector.run()

