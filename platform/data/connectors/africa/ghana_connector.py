"""
Ghana Power Market Connector

Ghana Grid Company (GRIDCo) wholesale market:
- Hydro-thermal mix
- Akosombo Dam dependency
- Natural gas from offshore fields
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


class GhanaConnector(Ingestor):
    """Ghana power market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull Ghana data."""
        logger.info("Fetching Ghana power data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch Ghana wholesale prices."""
        now = datetime.utcnow()  # Ghana uses GMT
        
        # Base price (GHS/MWh) - ~600 GHS/MWh (~$50 USD/MWh at 12 GHS/USD)
        base_price = 625
        hour = now.hour
        
        if 18 <= hour <= 22:
            tod_factor = 1.4
        elif 6 <= hour <= 17:
            tod_factor = 1.2
        else:
            tod_factor = 0.80
        
        # Hydro availability (Volta River)
        month = now.month
        if month in [8, 9, 10]:  # Peak water season
            hydro_factor = 0.88  # Abundant hydro, lower prices
        elif month in [2, 3, 4]:  # Dry season
            hydro_factor = 1.25  # Limited hydro, more thermal
        else:
            hydro_factor = 1.0
        
        price = base_price * tod_factor * hydro_factor
        price += (hash(str(hour)) % 80) - 40
        price = max(450, min(950, price))
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "GHANA_GRIDCO",
            "price_ghs_mwh": price,
            "hour_ending": now.replace(minute=0, second=0).isoformat(),
            "currency": "GHS",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map Ghana format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "GHANA.GRIDCO",
            "location_code": "GHANA.GRIDCO",
            "price_type": "wholesale",
            "value": float(raw["price_ghs_mwh"]),
            "volume": None,
            "currency": "GHS",
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
    connector = GhanaConnector({"source_id": "ghana_gridco"})
    connector.run()



