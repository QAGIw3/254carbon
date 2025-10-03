"""
SPP (Southwest Power Pool) Connector

Ingests Real-Time and Day-Ahead LMP, Integrated Markets prices,
and Operating Reserve data from SPP.
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

import requests
from kafka import KafkaProducer
import json

from .base import Ingestor

logger = logging.getLogger(__name__)


class SPPConnector(Ingestor):
    """SPP market data connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://marketplace.spp.org/web/api")
        self.api_key = config.get("api_key")
        self.market_type = config.get("market_type", "RT")  # RT, DA, IM
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover SPP data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "rt_lmp",
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "5min",
                    "nodes": "~2000",
                },
                {
                    "name": "da_lmp",
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "1day",
                    "nodes": "~2000",
                },
                {
                    "name": "integrated_markets",
                    "market": "power",
                    "product": "lmp",
                    "description": "Integrated Marketplace prices",
                },
                {
                    "name": "operating_reserves",
                    "market": "power",
                    "product": "ancillary",
                    "description": "Regulation and Spinning Reserve",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull data from SPP Marketplace API.
        
        SPP provides data through their Marketplace API.
        """
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching SPP {self.market_type} data since {last_time}")
        
        if self.market_type == "RT":
            yield from self._fetch_realtime_lmp()
        elif self.market_type == "DA":
            yield from self._fetch_dayahead_lmp()
        elif self.market_type == "IM":
            yield from self._fetch_integrated_markets()
        elif self.market_type == "OR":
            yield from self._fetch_operating_reserves()
    
    def _fetch_realtime_lmp(self) -> Iterator[Dict[str, Any]]:
        """Fetch Real-Time LMP (5-minute intervals)."""
        # Mock data - in production would call SPP API
        # Example: GET /settlement-location-prices/rtbm
        
        # Sample SPP settlement locations
        settlement_locations = [
            "SPP.NODE.GEN_001",
            "SPP.NODE.GEN_002",
            "SPP.HUB.NORTH",
            "SPP.HUB.SOUTH",
            "SPP.ZONE.KCPL",
            "SPP.ZONE.SWPS",
            "SPP.ZONE.WFEC",
            "SPP.ZONE.OKGE",
        ]
        
        # Current 5-min interval
        now = datetime.utcnow()
        interval_ending = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        
        for location in settlement_locations:
            # Mock LMP with components
            base_price = 35.0
            energy = base_price + (hash(location) % 20) - 10
            congestion = max(-5, min(5, (hash(location + "cong") % 10) - 5))
            loss = max(0, (hash(location + "loss") % 3) - 1)
            
            lmp = energy + congestion + loss
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "settlement_location": location,
                "lmp": lmp,
                "energy_component": energy,
                "congestion_component": congestion,
                "loss_component": loss,
                "interval_ending": interval_ending.isoformat(),
                "market": "RTBM",  # Real-Time Balancing Market
            }
    
    def _fetch_dayahead_lmp(self) -> Iterator[Dict[str, Any]]:
        """Fetch Day-Ahead LMP."""
        settlement_locations = [
            "SPP.HUB.NORTH",
            "SPP.HUB.SOUTH",
            "SPP.ZONE.KCPL",
            "SPP.ZONE.SWPS",
        ]
        
        # DA market for tomorrow
        tomorrow = datetime.utcnow() + timedelta(days=1)
        
        for hour in range(24):
            interval = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            for location in settlement_locations:
                # Mock DA LMP
                base_price = 32.0
                hourly_factor = 1 + 0.3 * (abs(hour - 14) / 14)  # Peak at 2pm
                
                energy = base_price * hourly_factor
                congestion = (hash(location + str(hour)) % 8) - 4
                loss = max(0, (hash(location) % 2))
                
                lmp = energy + congestion + loss
                
                yield {
                    "timestamp": datetime.utcnow().isoformat(),
                    "settlement_location": location,
                    "lmp": lmp,
                    "energy_component": energy,
                    "congestion_component": congestion,
                    "loss_component": loss,
                    "interval_ending": interval.isoformat(),
                    "market": "DAM",  # Day-Ahead Market
                }
    
    def _fetch_integrated_markets(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Integrated Marketplace prices.
        
        SPP's Integrated Marketplace combines RT and DA operations.
        """
        now = datetime.utcnow()
        
        # IM hubs
        im_hubs = [
            "SPP.IM.NORTH_HUB",
            "SPP.IM.SOUTH_HUB",
            "SPP.IM.INTERFACE_EAST",
            "SPP.IM.INTERFACE_WEST",
        ]
        
        for hub in im_hubs:
            # Mock IM price
            base_price = 36.0
            price = base_price + (hash(hub) % 15) - 7
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "hub_id": hub,
                "price": price,
                "market": "IM",
                "interval_ending": now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0).isoformat(),
            }
    
    def _fetch_operating_reserves(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Operating Reserve prices.
        
        Includes Regulation Up/Down and Spinning Reserve.
        """
        now = datetime.utcnow()
        interval_ending = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        
        # Reserve products
        reserve_products = [
            {"type": "REG_UP", "typical_price": 12.0},
            {"type": "REG_DOWN", "typical_price": 8.0},
            {"type": "SPIN", "typical_price": 10.0},
            {"type": "SUPP", "typical_price": 6.0},
        ]
        
        for product in reserve_products:
            # Mock reserve price
            price = product["typical_price"] + (hash(product["type"]) % 5) - 2
            
            # Mock MW cleared
            mw_cleared = 500 + (hash(product["type"] + str(now.hour)) % 300)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "product": product["type"],
                "price": max(0, price),
                "mw_cleared": mw_cleared,
                "interval_ending": interval_ending.isoformat(),
                "market": "OR",
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map SPP format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product type
        if "lmp" in raw:
            product = "lmp"
            value = raw["lmp"]
            location = raw["settlement_location"]
        elif "hub_id" in raw:
            product = "lmp"
            value = raw["price"]
            location = raw["hub_id"]
        elif "product" in raw:
            product = "ancillary"
            value = raw["price"]
            location = f"SPP.OR.{raw['product']}"
        else:
            product = "unknown"
            value = 0
            location = "SPP.UNKNOWN"
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "trade",
            "value": float(value),
            "volume": raw.get("mw_cleared"),
            "currency": "USD",
            "unit": "MWh" if product == "lmp" else "MW",
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
        logger.info(f"Emitted {count} SPP events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"SPP checkpoint saved: {state}")


if __name__ == "__main__":
    # Test SPP connector
    configs = [
        {
            "source_id": "spp_rtbm",
            "market_type": "RT",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "spp_dam",
            "market_type": "DA",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "spp_reserves",
            "market_type": "OR",
            "kafka_topic": "power.ancillary.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = SPPConnector(config)
        connector.run()

