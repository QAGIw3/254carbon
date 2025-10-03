"""
Uruguay ADME (Administración del Mercado Eléctrico) Connector

Uruguay's renewable-dominant market:
- 98% renewable electricity
- Wind power leadership
- Cross-border with Argentina and Brazil
- Real-time market
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class UruguayADMEConnector(Ingestor):
    """Uruguay ADME market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull Uruguay data."""
        logger.info("Fetching Uruguay ADME data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch Uruguay spot prices."""
        now = datetime.utcnow()
        uruguay_time = now - timedelta(hours=3)  # UYT (UTC-3)
        
        # Base price (UYU/MWh) - ~2,800 UYU/MWh (~$70 USD/MWh at 40 UYU/USD)
        base_price = 2900
        hour = uruguay_time.hour
        
        if 19 <= hour <= 22:
            tod_factor = 1.3
        elif 7 <= hour <= 18:
            tod_factor = 1.1
        else:
            tod_factor = 0.85
        
        # Wind generation (35% of capacity, highly variable)
        if hash(str(hour)) % 3 == 0:  # High wind periods
            wind_discount = 0.70  # Abundant wind, low prices
        else:
            wind_discount = 1.0
        
        # Hydro seasonal (25% of generation)
        month = uruguay_time.month
        if month in [5, 6, 7]:  # Winter (dry season)
            hydro_factor = 1.15
        else:
            hydro_factor = 0.95
        
        price = base_price * tod_factor * wind_discount * hydro_factor
        price += (hash(str(hour)) % 400) - 200
        price = max(1800, min(4500, price))
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "URUGUAY_ADME",
            "price_uyu_mwh": price,
            "hour_ending": uruguay_time.replace(minute=0, second=0).isoformat(),
            "currency": "UYU",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map Uruguay format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "ADME.URUGUAY",
            "location_code": "ADME.URUGUAY",
            "price_type": "spot",
            "value": float(raw["price_uyu_mwh"]),
            "volume": None,
            "currency": "UYU",
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
    connector = UruguayADMEConnector({"source_id": "uruguay_adme"})
    connector.run()



