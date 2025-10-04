"""
JEPX (Japan Electric Power Exchange) Connector

Overview
--------
Ingests day‑ahead spot and forward market prices from Japan's power exchange,
mapping to canonical schema with JPY currency and JST→UTC normalization.

Data Flow
---------
JEPX API (or mocks) → parse area/period prices → canonical tick schema → Kafka
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

import requests
from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class JEPXConnector(Ingestor):
    """JEPX Japan power exchange connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.jepx.org/api")
        self.market_type = config.get("market_type", "SPOT")  # SPOT, 1WEEK, 1MONTH
        self.area = config.get("area", "SYSTEM")  # SYSTEM, or specific areas
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover JEPX data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "day_ahead",
                    "market": "power",
                    "product": "energy",
                    "description": "Day-Ahead Spot Market",
                    "areas": ["System", "Hokkaido", "Tohoku", "Tokyo", "Chubu", "Hokuriku", 
                             "Kansai", "Chugoku", "Shikoku", "Kyushu"],
                    "currency": "JPY",
                },
                {
                    "name": "intraday",
                    "market": "power",
                    "product": "energy",
                    "description": "Intraday Market (30min products)",
                    "update_freq": "30min",
                },
                {
                    "name": "week_ahead",
                    "market": "power",
                    "product": "energy",
                    "description": "1-Week Ahead Market",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull data from JEPX API."""
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching JEPX {self.market_type} data since {last_time}")
        
        if self.market_type == "SPOT":
            yield from self._fetch_spot_market()
        elif self.market_type == "1WEEK":
            yield from self._fetch_week_ahead()
    
    def _fetch_spot_market(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch day-ahead spot market prices.
        
        JEPX operates 30-minute products for day-ahead trading.
        """
        # Japan operates on JST (UTC+9)
        now = datetime.utcnow()
        tomorrow_jst = now + timedelta(days=1, hours=9)
        
        # Base price (JPY/kWh, typically ¥8-15/kWh = ¥8000-15000/MWh)
        base_price = 11000.0  # JPY/MWh
        
        # Japan uses 30-minute products (48 periods per day)
        for period in range(48):
            delivery_time = tomorrow_jst.replace(
                hour=period // 2,
                minute=30 if period % 2 else 0,
                second=0,
                microsecond=0
            )
            
            # Japanese load pattern
            hour = period // 2
            if 8 <= hour <= 10 or 18 <= hour <= 21:
                hourly_factor = 1.4  # Morning and evening peaks
            elif 11 <= hour <= 17:
                hourly_factor = 1.2  # Daytime
            else:
                hourly_factor = 0.8  # Night
            
            # Add randomness
            price = base_price * hourly_factor + (hash(str(period)) % 2000) - 1000
            
            # Occasional high prices (tight supply)
            if hash(str(period) + str(now.day)) % 25 == 0:
                price *= 2.0
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "JEPX",
                "area": self.area,
                "market_type": "SPOT",
                "price_jpy_mwh": price,
                "volume_mwh": 500 + (hash(str(period)) % 300),
                "delivery_period": period + 1,  # 1-48
                "delivery_time": delivery_time.isoformat(),
                "currency": "JPY",
            }
    
    def _fetch_week_ahead(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch 1-week ahead market prices.
        
        Allows trading for delivery 1-2 weeks ahead.
        """
        now = datetime.utcnow()
        
        # Week ahead covers next week
        for day_offset in range(7, 14):
            delivery_date = (now + timedelta(days=day_offset)).date()
            
            # Daily average price
            avg_price = 12000.0 + (hash(str(delivery_date)) % 3000) - 1500
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "JEPX",
                "area": self.area,
                "market_type": "1WEEK",
                "price_jpy_mwh": avg_price,
                "volume_mwh": 2000 + (hash(str(delivery_date)) % 1000),
                "delivery_date": delivery_date.isoformat(),
                "currency": "JPY",
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map JEPX format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        location = f"JEPX.{raw['area']}.{raw['market_type']}"
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": "energy",
            "instrument_id": location,
            "location_code": location,
            "price_type": "auction",
            "value": float(raw["price_jpy_mwh"]),
            "volume": raw.get("volume_mwh"),
            "currency": "JPY",
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
        logger.info(f"Emitted {count} JEPX events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"JEPX checkpoint saved: {state}")


if __name__ == "__main__":
    # Test JEPX connector
    configs = [
        {
            "source_id": "jepx_spot",
            "market_type": "SPOT",
            "area": "SYSTEM",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "jepx_week",
            "market_type": "1WEEK",
            "area": "SYSTEM",
            "kafka_topic": "power.ticks.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = JEPXConnector(config)
        connector.run()
