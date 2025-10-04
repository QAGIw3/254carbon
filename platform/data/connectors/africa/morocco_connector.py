"""
Morocco Power Market Connector

Integrates with Morocco's renewable‑focused power system:
- Noor solar complex (world's largest CSP)
- Wind energy expansion
- Spain interconnection (exports to Europe)
- Natural gas imports

Data Flow
---------
ONEE feeds (or mocks) → normalize (MAD/MWh, local time→UTC) → canonical schema → Kafka
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class MoroccoConnector(Ingestor):
    """Morocco power market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.onee.ma/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover Morocco market streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "system_prices",
                    "description": "ONEE system prices",
                    "currency": "MAD",
                },
                {
                    "name": "renewable_generation",
                    "description": "Solar CSP and wind output",
                },
                {
                    "name": "spain_interconnection",
                    "description": "Cross-border flows to/from Spain",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull Morocco market data."""
        logger.info("Fetching Morocco power data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch Morocco system prices."""
        now = datetime.utcnow()
        morocco_time = now  # Morocco uses UTC in winter, UTC+1 in summer
        
        # Base price (MAD/MWh) - ~900 MAD/MWh (~$90 USD/MWh at 10 MAD/USD)
        base_price = 900
        
        hour = morocco_time.hour
        
        # Time of day
        if 19 <= hour <= 23:
            tod_factor = 1.4
        elif 7 <= hour <= 18:
            tod_factor = 1.2
        else:
            tod_factor = 0.80
        
        # Solar CSP provides evening capacity
        # Noor CSP has molten salt storage
        if 18 <= hour <= 23:
            solar_storage_benefit = 0.92  # CSP displaces expensive peakers
        else:
            solar_storage_benefit = 1.0
        
        # Wind generation (strong Atlantic winds)
        if hash(str(hour)) % 3 == 0:  # High wind periods
            wind_discount = 0.88
        else:
            wind_discount = 1.0
        
        price = base_price * tod_factor * solar_storage_benefit * wind_discount
        price += (hash(str(hour)) % 120) - 60
        price = max(650, min(1400, price))
        
        # Load (MW)
        load = 5500 + (hash(str(hour)) % 1000)
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "MOROCCO_ONEE",
            "price_mad_mwh": price,
            "hour_ending": morocco_time.replace(minute=0, second=0).isoformat(),
            "load_mw": load,
            "currency": "MAD",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map Morocco format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "MOROCCO.ONEE",
            "location_code": "MOROCCO.ONEE",
            "price_type": "system",
            "value": float(raw["price_mad_mwh"]),
            "volume": raw.get("load_mw"),
            "currency": "MAD",
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
        logger.info(f"Emitted {count} Morocco events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = MoroccoConnector({"source_id": "morocco_onee"})
    connector.run()
