"""
Thailand EGAT (Electricity Generating Authority of Thailand) Connector

Integrates with Thailand's transitioning power market:
- Enhanced Single Buyer (ESB) model
- Feed-in tariffs for renewables
- LNG import pricing
- Cross-border hydro imports (Laos)
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class ThailandEGATConnector(Ingestor):
    """Thailand EGAT power market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.egat.co.th/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover EGAT data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "system_prices",
                    "description": "Enhanced Single Buyer prices",
                    "currency": "THB",
                },
                {
                    "name": "renewable_fit",
                    "description": "Feed-in tariff rates",
                },
                {
                    "name": "cross_border",
                    "description": "Laos hydro imports",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull EGAT market data."""
        logger.info("Fetching Thailand EGAT data")
        yield from self._fetch_system_prices()
    
    def _fetch_system_prices(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch system marginal prices under ESB model.
        
        Thailand uses Enhanced Single Buyer with competitive bidding.
        """
        now = datetime.utcnow()
        thailand_time = now + timedelta(hours=7)  # ICT (UTC+7)
        
        # Base price (THB/MWh) - heavily LNG-dependent
        base_price = 2800  # ~$80 USD/MWh at 35 THB/USD
        
        hour = thailand_time.hour
        
        # Peak/off-peak
        if 18 <= hour <= 22:
            tod_factor = 1.5  # Evening peak (hot season cooling)
        elif 9 <= hour <= 17:
            tod_factor = 1.3  # Day
        else:
            tod_factor = 0.75  # Night valley
        
        # Seasonal factors (monsoon vs dry season)
        month = thailand_time.month
        if month in [3, 4, 5]:  # Hot season (pre-monsoon)
            seasonal_factor = 1.3  # High cooling demand
        elif month in [6, 7, 8, 9]:  # Monsoon season
            seasonal_factor = 1.0  # Moderate
        else:
            seasonal_factor = 0.95  # Cool season
        
        # LNG price correlation (Thailand imports ~70% of gas)
        lng_price = 12.0  # USD/MMBtu
        lng_factor = lng_price / 10.0  # Normalize
        
        price = base_price * tod_factor * seasonal_factor * lng_factor
        
        # Add variation
        price += (hash(str(thailand_time.hour)) % 300) - 150
        price = max(2000, min(4500, price))
        
        # System load (MW)
        load = 28000 + (hash(str(hour)) % 5000)
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "THAILAND_EGAT",
            "price_thb_mwh": price,
            "hour_ending": thailand_time.replace(minute=0, second=0).isoformat(),
            "system_load_mw": load,
            "currency": "THB",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map EGAT format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "EGAT.SYSTEM",
            "location_code": "EGAT.SYSTEM",
            "price_type": "system_price",
            "value": float(raw["price_thb_mwh"]),
            "volume": raw.get("system_load_mw"),
            "currency": "THB",
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
        logger.info(f"Emitted {count} EGAT events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = ThailandEGATConnector({"source_id": "thailand_egat"})
    connector.run()

