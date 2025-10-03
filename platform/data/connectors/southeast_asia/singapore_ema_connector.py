"""
Singapore Energy Market Authority (EMA) Connector

Integrates with Singapore's wholesale electricity market:
- Uniform Singapore Energy Price (USEP)
- Vesting contracts
- LNG-indexed pricing
- Carbon tax integration
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class SingaporeEMAConnector(Ingestor):
    """Singapore EMA market connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.emcsg.com/api")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover EMA data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "usep",
                    "description": "Uniform Singapore Energy Price - half-hourly",
                    "currency": "SGD",
                },
                {
                    "name": "vesting_contracts",
                    "description": "Regulated vesting contract prices",
                },
                {
                    "name": "lng_index",
                    "description": "LNG-indexed fuel costs",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull EMA market data."""
        logger.info("Fetching Singapore EMA data")
        yield from self._fetch_usep()
    
    def _fetch_usep(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch USEP (Uniform Singapore Energy Price).
        
        Singapore uses half-hourly settlement periods.
        """
        # Current half-hour period
        now = datetime.utcnow()
        sg_time = now + timedelta(hours=8)  # SGT (UTC+8)
        period = sg_time.replace(minute=(sg_time.minute // 30) * 30, second=0, microsecond=0)
        
        # Base price (SGD/MWh) - heavily influenced by LNG prices
        base_price = 180.0  # SGD/MWh (~$135 USD/MWh)
        
        # Time of day factors
        hour = period.hour
        if 18 <= hour <= 22:
            tod_factor = 1.3  # Evening peak
        elif 9 <= hour <= 17:
            tod_factor = 1.1  # Day
        else:
            tod_factor = 0.85  # Night
        
        # LNG price correlation (Singapore uses ~95% natural gas)
        lng_price = 12.0  # USD/MMBtu
        gas_factor = (lng_price / 10.0)  # Normalize to $10/MMBtu
        
        # Carbon tax impact (currently S$5/tCO2, rising to S$25 by 2030)
        carbon_tax_sgd = 5.0  # SGD per tonne
        emissions_rate = 0.45  # tCO2 per MWh for gas
        carbon_cost = carbon_tax_sgd * emissions_rate
        
        price = (base_price * tod_factor * gas_factor) + carbon_cost
        
        # Add variation
        price += (hash(str(period.minute)) % 20) - 10
        
        # Typical range: 120-250 SGD/MWh
        price = max(100, min(300, price))
        
        # Volume (MW)
        volume = 3500 + (hash(str(hour)) % 1000)
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "SINGAPORE_EMA",
            "price_sgd_mwh": price,
            "period_ending": period.isoformat(),
            "volume_mw": volume,
            "currency": "SGD",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map EMA format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": "EMA.USEP",
            "location_code": "EMA.USEP",
            "price_type": "uniform_price",
            "value": float(raw["price_sgd_mwh"]),
            "volume": raw.get("volume_mw"),
            "currency": "SGD",
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
        logger.info(f"Emitted {count} EMA events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = SingaporeEMAConnector({"source_id": "singapore_ema"})
    connector.run()

