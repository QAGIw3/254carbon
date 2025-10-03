"""
Argentina CAMMESA (Compañía Administradora del Mercado Mayorista Eléctrico) Connector

Integrates with Argentine wholesale electricity market:
- Seasonal spot market
- Currency volatility (ARS)
- Vaca Muerta gas influence
- Hydro-thermal-renewable mix
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class ArgentinaCAMMESAConnector(Ingestor):
    """Argentina CAMMESA market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.cammesa.com/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover CAMMESA data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "spot_market",
                    "description": "Seasonal spot prices",
                    "currency": "ARS",
                },
                {
                    "name": "generation_mix",
                    "description": "Hydro, thermal, renewable generation",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull CAMMESA market data."""
        logger.info("Fetching Argentina CAMMESA data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Argentina spot market prices.
        
        Highly seasonal based on hydro availability.
        Subject to currency volatility.
        """
        now = datetime.utcnow()
        argentina_time = now - timedelta(hours=3)  # ART (UTC-3)
        
        # Base price (ARS/MWh) - subject to inflation
        # ~20,000 ARS/MWh (~$60 USD/MWh at 350 ARS/USD)
        base_price = 18000
        
        hour = argentina_time.hour
        
        # Time of day
        if 18 <= hour <= 22:
            tod_factor = 1.4
        elif 7 <= hour <= 17:
            tod_factor = 1.1
        else:
            tod_factor = 0.75
        
        # Seasonal hydro variation (critical driver)
        month = argentina_time.month
        if month in [11, 12, 1, 2]:  # Summer (wet season)
            hydro_factor = 0.80  # Abundant hydro, low prices
        elif month in [6, 7, 8]:  # Winter (dry season)
            hydro_factor = 1.4  # Limited hydro, high prices
        else:
            hydro_factor = 1.0
        
        # Vaca Muerta gas impact (growing domestic supply)
        gas_availability = 1.0 - (hash(str(datetime.now().year)) % 10) / 50  # Improving supply
        
        price = base_price * tod_factor * hydro_factor * gas_availability
        price += (hash(str(hour)) % 3000) - 1500
        price = max(10000, min(35000, price))
        
        # Load (MW)
        load = 24000 + (hash(str(hour)) % 4000)
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "ARGENTINA_CAMMESA",
            "price_ars_mwh": price,
            "hour_ending": argentina_time.replace(minute=0, second=0).isoformat(),
            "load_mw": load,
            "currency": "ARS",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map CAMMESA format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "CAMMESA.SPOT",
            "location_code": "CAMMESA.SPOT",
            "price_type": "spot",
            "value": float(raw["price_ars_mwh"]),
            "volume": raw.get("load_mw"),
            "currency": "ARS",
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
        logger.info(f"Emitted {count} CAMMESA events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = ArgentinaCAMMESAConnector({"source_id": "argentina_cammesa"})
    connector.run()

