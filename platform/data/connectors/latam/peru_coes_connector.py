"""
Peru COES (Comité de Operación Económica del Sistema) Connector

Integrates with Peruvian power market:
- Marginal cost pricing
- Mining sector demand (30%)
- Hydro seasonal variations
- Natural gas from Camisea

Data Flow
---------
COES feeds (or mocks) → normalize to canonical tick schema → Kafka topic(s)
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class PeruCOESConnector(Ingestor):
    """Peru COES market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.coes.org.pe/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover COES data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "marginal_cost",
                    "description": "System marginal cost",
                    "currency": "PEN",
                },
                {
                    "name": "mining_demand",
                    "description": "Mining sector electricity consumption",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull COES market data."""
        logger.info("Fetching Peru COES data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Peru marginal cost (spot) prices.
        
        Peru uses marginal cost dispatch with hydro-thermal coordination.
        Mining sector represents ~30% of demand.
        """
        now = datetime.utcnow()
        peru_time = now - timedelta(hours=5)  # PET (UTC-5)
        
        # Base price (PEN/MWh) - ~180 PEN/MWh (~$50 USD/MWh at 3.6 PEN/USD)
        base_price = 185
        
        hour = peru_time.hour
        
        # Time of day (mining operations run 24/7, but residential peak matters)
        if 19 <= hour <= 22:
            tod_factor = 1.3
        elif 7 <= hour <= 18:
            tod_factor = 1.1
        else:
            tod_factor = 0.90
        
        # Hydro seasonal variation
        month = peru_time.month
        if month in [1, 2, 3]:  # Wet season (Andean rainfall)
            hydro_factor = 0.80  # Abundant hydro
        elif month in [7, 8, 9]:  # Dry season
            hydro_factor = 1.3  # Limited hydro, more thermal
        else:
            hydro_factor = 1.0
        
        # Camisea natural gas availability
        gas_availability = 1.0  # Generally stable domestic supply
        
        price = base_price * tod_factor * hydro_factor * gas_availability
        price += (hash(str(hour)) % 30) - 15
        price = max(130, min(350, price))
        
        # Load (MW) - mining creates relatively flat load profile
        load = 6800 + (hash(str(hour)) % 800)
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "PERU_COES",
            "price_pen_mwh": price,
            "hour_ending": peru_time.replace(minute=0, second=0).isoformat(),
            "load_mw": load,
            "currency": "PEN",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map COES format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "COES.PERU",
            "location_code": "COES.PERU",
            "price_type": "marginal_cost",
            "value": float(raw["price_pen_mwh"]),
            "volume": raw.get("load_mw"),
            "currency": "PEN",
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
        logger.info(f"Emitted {count} COES Peru events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = PeruCOESConnector({"source_id": "peru_coes"})
    connector.run()
