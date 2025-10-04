"""
Colombia XM Connector

Integrates with Colombian wholesale electricity market:
- Spot market (Bolsa de Energía)
- Hydro‑thermal optimization
- El Niño/La Niña impacts
- Reliability charge

Data Flow
---------
XM feeds (or mocks) → normalize to canonical tick schema → Kafka topic(s)
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class ColombiaXMConnector(Ingestor):
    """Colombia XM market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.xm.com.co/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover XM data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "spot_market",
                    "description": "Bolsa de Energía hourly prices",
                    "currency": "COP",
                },
                {
                    "name": "hydro_levels",
                    "description": "Reservoir levels and El Niño impact",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull XM market data."""
        logger.info("Fetching Colombia XM data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Colombia spot market prices.
        
        Heavily influenced by hydro availability and El Niño/La Niña.
        """
        now = datetime.utcnow()
        colombia_time = now - timedelta(hours=5)  # COT (UTC-5)
        
        # Base price (COP/MWh) - ~200,000 COP/MWh (~$50 USD/MWh at 4,000 COP/USD)
        base_price = 210000
        
        hour = colombia_time.hour
        
        # Time of day
        if 18 <= hour <= 21:
            tod_factor = 1.3
        elif 6 <= hour <= 17:
            tod_factor = 1.1
        else:
            tod_factor = 0.80
        
        # El Niño/La Niña impact (critical for Colombia ~70% hydro)
        month = colombia_time.month
        year = colombia_time.year
        
        # Mock El Niño/La Niña cycle
        if year % 4 == 0:  # El Niño year (dry)
            enso_factor = 1.6  # Much higher prices
        elif year % 4 == 2:  # La Niña year (wet)
            enso_factor = 0.75  # Lower prices
        else:
            enso_factor = 1.0  # Neutral
        
        # Additional seasonal variation
        if month in [12, 1, 2]:  # Dry season
            seasonal_factor = 1.2
        elif month in [4, 5, 10, 11]:  # Rainy seasons
            seasonal_factor = 0.85
        else:
            seasonal_factor = 1.0
        
        combined_hydro_factor = enso_factor * seasonal_factor
        
        price = base_price * tod_factor * combined_hydro_factor
        price += (hash(str(hour)) % 40000) - 20000
        price = max(120000, min(550000, price))  # Wide range due to hydro
        
        # Load (MW)
        load = 9500 + (hash(str(hour)) % 1500)
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "COLOMBIA_XM",
            "price_cop_mwh": price,
            "hour_ending": colombia_time.replace(minute=0, second=0).isoformat(),
            "load_mw": load,
            "currency": "COP",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map XM format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "XM.COLOMBIA",
            "location_code": "XM.COLOMBIA",
            "price_type": "spot",
            "value": float(raw["price_cop_mwh"]),
            "volume": raw.get("load_mw"),
            "currency": "COP",
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
        logger.info(f"Emitted {count} XM Colombia events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = ColombiaXMConnector({"source_id": "colombia_xm"})
    connector.run()
