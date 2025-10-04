"""
Chile CEN (Coordinador Eléctrico Nacional) Connector

Chilean power market with unified national system:
- Marginal cost pricing
- Copper mining demand (30%)
- Solar Atacama Desert
- Hydro‑thermal coordination

Data Flow
---------
CEN feeds (or mocks) → normalize to canonical tick schema → Kafka topic(s)
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class ChileCENConnector(Ingestor):
    """Chile CEN market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull Chile data."""
        logger.info("Fetching Chile CEN data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch Chile marginal cost prices."""
        now = datetime.utcnow()
        chile_time = now - timedelta(hours=3)  # CLT (UTC-3)
        
        # Base price (CLP/MWh) - ~65,000 CLP/MWh (~$75 USD/MWh at 850 CLP/USD)
        base_price = 67000
        hour = chile_time.hour
        
        if 19 <= hour <= 23:
            tod_factor = 1.4
        elif 7 <= hour <= 18:
            tod_factor = 1.2
        else:
            tod_factor = 0.80
        
        # Copper mining (24/7 operations) flattens load curve
        mining_factor = 0.95
        
        # Solar Atacama (world's best resource)
        if 10 <= hour <= 16:  # Mid-day solar abundance
            solar_discount = 0.75  # Significant price suppression
        else:
            solar_discount = 1.0
        
        price = base_price * tod_factor * mining_factor * solar_discount
        price += (hash(str(hour)) % 6000) - 3000
        price = max(45000, min(95000, price))
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "CHILE_CEN",
            "price_clp_mwh": price,
            "hour_ending": chile_time.replace(minute=0, second=0).isoformat(),
            "currency": "CLP",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map Chile format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "CEN.CHILE",
            "location_code": "CEN.CHILE",
            "price_type": "marginal_cost",
            "value": float(raw["price_clp_mwh"]),
            "volume": None,
            "currency": "CLP",
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
    connector = ChileCENConnector({"source_id": "chile_cen"})
    connector.run()


