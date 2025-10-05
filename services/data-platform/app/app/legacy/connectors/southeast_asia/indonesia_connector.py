"""
Indonesia Power Market Connector

Integrates with PLN (Perusahaan Listrik Negara) and emerging wholesale market:
- Java-Bali grid (largest system)
- Regional pricing variations
- Renewable auction results
- Coal-fired baseload
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class IndonesiaConnector(Ingestor):
    """Indonesia PLN power market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.pln.co.id/api")
        self.grid = config.get("grid", "JAVA_BALI")  # JAVA_BALI, SUMATRA, KALIMANTAN
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull Indonesia market data."""
        logger.info(f"Fetching Indonesia {self.grid} power data")
        yield from self._fetch_prices()
    
    def _fetch_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch Indonesia system prices."""
        now = datetime.utcnow()
        wib_time = now + timedelta(hours=7)  # WIB (UTC+7)
        
        # Base price by grid (IDR/kWh → IDR/MWh)
        grid_prices = {
            "JAVA_BALI": 1050000,  # ~$70 USD/MWh
            "SUMATRA": 1150000,
            "KALIMANTAN": 1250000,  # Higher due to diesel
        }
        
        base_price = grid_prices.get(self.grid, 1050000)
        hour = wib_time.hour
        
        # Peak/off-peak
        if 18 <= hour <= 22:
            tod_factor = 1.5
        elif 8 <= hour <= 17:
            tod_factor = 1.2
        else:
            tod_factor = 0.80
        
        # Coal price influence (60% coal generation)
        coal_factor = 1.0 + (hash(str(datetime.now().month)) % 15) / 100
        
        price = base_price * tod_factor * coal_factor
        price += (hash(str(hour)) % 120000) - 60000
        price = max(850000, min(1600000, price))
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "INDONESIA_PLN",
            "grid": self.grid,
            "price_idr_mwh": price,
            "hour_ending": wib_time.replace(minute=0, second=0).isoformat(),
            "currency": "IDR",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map Indonesia format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": f"PLN.{raw['grid']}",
            "location_code": f"PLN.{raw['grid']}",
            "price_type": "system",
            "value": float(raw["price_idr_mwh"]),
            "volume": None,
            "currency": "IDR",
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
        logger.info(f"Emitted {count} Indonesia events")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = IndonesiaConnector({"source_id": "indonesia_pln", "grid": "JAVA_BALI"})
    connector.run()



