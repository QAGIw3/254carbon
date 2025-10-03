"""
PJM Interconnection Connector

Ingests real-time and day-ahead LMP data, capacity market data,
and ancillary services from PJM.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator, Dict, Any, Optional
import time

import requests
from kafka import KafkaProducer
import json

from .base import Ingestor

logger = logging.getLogger(__name__)


class PJMConnector(Ingestor):
    """PJM real-time, day-ahead, capacity, and ancillary services connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://api.pjm.com/api/v1")
        self.api_key = config.get("api_key")
        self.market_type = config.get("market_type", "RT")  # RT, DA, CAPACITY, AS
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover PJM pricing nodes and products."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "nodal_lmp_rt",
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "5min",
                    "nodes": "~11000",  # PJM has ~11,000 pricing nodes
                },
                {
                    "name": "nodal_lmp_da",
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "1hour",
                },
                {
                    "name": "capacity_prices",
                    "market": "power",
                    "product": "capacity",
                    "update_freq": "daily",
                },
                {
                    "name": "ancillary_services",
                    "market": "power",
                    "product": "ancillary",
                    "update_freq": "5min",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull data from PJM API.
        
        Real implementation would use actual PJM OASIS API with authentication.
        """
        last_checkpoint = self.load_checkpoint()
        last_time = self._resolve_last_time(last_checkpoint)
        
        logger.info(f"Fetching PJM {self.market_type} data since {last_time}")
        
        if self.market_type == "RT":
            yield from self._fetch_realtime_lmp(last_time)
        elif self.market_type == "DA":
            yield from self._fetch_dayahead_lmp(last_time)
        elif self.market_type == "CAPACITY":
            yield from self._fetch_capacity_prices()
        elif self.market_type == "AS":
            yield from self._fetch_ancillary_services()
    
    def _fetch_realtime_lmp(self, last_time: datetime) -> Iterator[Dict[str, Any]]:
        """Fetch real-time LMP data."""
        # Mock data - in production would call PJM API
        # Example: GET /v1/rt_hrl_lmps
        
        # Sample PJM zones and hubs
        locations = [
            "PJM.HUB.WEST",
            "PJM.HUB.AEP",
            "PJM.ZONE.AE",
            "PJM.ZONE.APS",
            "PJM.ZONE.ATSI",
            "PJM.ZONE.BC",
            "PJM.ZONE.COMED",
            "PJM.ZONE.DAY",
            "PJM.ZONE.DEOK",
            "PJM.ZONE.DOM",
        ]
        
        for location in locations:
            # Mock LMP components
            energy = 35.0 + (hash(location) % 20)
            congestion = -2.5 + (hash(location + "cong") % 10)
            loss = 0.5 + (hash(location + "loss") % 3) / 10
            
            event_time = datetime.now(timezone.utc)
            yield {
                "timestamp": event_time.isoformat(),
                "node_id": location,
                "lmp": energy + congestion + loss,
                "energy_component": energy,
                "congestion_component": congestion,
                "loss_component": loss,
                "market": "RT",
                "interval_ending": event_time.isoformat(),
            }
    
    def _fetch_dayahead_lmp(self, last_time: datetime) -> Iterator[Dict[str, Any]]:
        """Fetch day-ahead LMP data."""
        # Mock data - in production would call PJM API
        # Example: GET /v1/da_hrl_lmps
        
        locations = ["PJM.HUB.WEST", "PJM.HUB.AEP", "PJM.ZONE.COMED"]
        
        # Generate hourly prices for next day
        for hour in range(24):
            for location in locations:
                energy = 40.0 + (hash(f"{location}{hour}") % 25)
                congestion = -1.0 + (hash(f"{location}{hour}cong") % 8)
                loss = 0.8
                
                event_time = datetime.now(timezone.utc)
                yield {
                    "timestamp": event_time.isoformat(),
                    "node_id": location,
                    "lmp": energy + congestion + loss,
                    "energy_component": energy,
                    "congestion_component": congestion,
                    "loss_component": loss,
                    "market": "DA",
                    "interval_ending": (
                        event_time + timedelta(days=1, hours=hour)
                    ).isoformat(),
                    "hour_ending": hour + 1,
                }
    
    def _fetch_capacity_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch capacity auction results."""
        # Mock capacity prices by zone
        # In production: GET /v1/capacity_market_results
        
        zones = {
            "RTO": 140.00,
            "EMAAC": 167.00,
            "SWMAAC": 140.00,
            "MAAC": 140.00,
            "DPL_SOUTH": 140.00,
        }
        
        delivery_year = datetime.utcnow().year + 3  # BRA is 3 years forward
        
        for zone, price in zones.items():
            yield {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "zone": f"PJM.{zone}",
                "product": "capacity",
                "clearing_price": price,
                "delivery_year": delivery_year,
                "auction": f"BRA_{delivery_year-3}/{delivery_year-2}",
            }
    
    def _fetch_ancillary_services(self) -> Iterator[Dict[str, Any]]:
        """Fetch ancillary services prices."""
        # Regulation, spinning reserve, etc.
        # In production: GET /v1/ancillary_services
        
        products = {
            "reg_up": 12.50,
            "reg_down": 8.00,
            "sync_reserve": 3.50,
            "nonsync_reserve": 2.00,
        }
        
        for product, price in products.items():
            yield {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "product": product,
                "market": "AS",
                "clearing_price": price + (hash(product) % 5) / 2,
                "interval_ending": datetime.utcnow().isoformat(),
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map PJM format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product type
        if "lmp" in raw:
            product = "lmp"
            value = raw["lmp"]
            location = raw["node_id"]
        elif raw.get("product") == "capacity":
            product = "capacity"
            value = raw["clearing_price"]
            location = raw["zone"]
        else:
            product = raw.get("product", "ancillary")
            value = raw["clearing_price"]
            location = "PJM.RTO"
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "settle" if raw.get("market") == "DA" else "trade",
            "value": float(value),
            "volume": None,
            "currency": "USD",
            "unit": "MWh" if product in ["lmp", "ancillary"] else "MW-day",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }

    def _resolve_last_time(self, checkpoint: Optional[Dict[str, Any]]) -> datetime:
        """Determine the timestamp to resume from when fetching PJM data."""
        if not checkpoint:
            return datetime.now(timezone.utc) - timedelta(hours=1)

        last_event_time = checkpoint.get("last_event_time")

        if last_event_time is None:
            return datetime.now(timezone.utc) - timedelta(hours=1)

        if isinstance(last_event_time, (int, float)):
            return datetime.fromtimestamp(last_event_time / 1000, tz=timezone.utc)

        if isinstance(last_event_time, str):
            try:
                dt = datetime.fromisoformat(last_event_time)
                return dt.astimezone(timezone.utc)
            except ValueError:
                logger.warning("Invalid last_event_time in checkpoint; defaulting to 1 hour lookback")
                return datetime.now(timezone.utc) - timedelta(hours=1)

        if isinstance(last_event_time, datetime):
            return last_event_time.astimezone(timezone.utc)

        logger.warning("Unsupported last_event_time type; defaulting to 1 hour lookback")
        return datetime.now(timezone.utc) - timedelta(hours=1)
    
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
        logger.info(f"Emitted {count} PJM events to {self.kafka_topic}")
        return count
    


if __name__ == "__main__":
    # Test PJM connector
    configs = [
        {
            "source_id": "pjm_rt_lmp",
            "market_type": "RT",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "pjm_da_lmp",
            "market_type": "DA",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "pjm_capacity",
            "market_type": "CAPACITY",
            "kafka_topic": "power.capacity.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = PJMConnector(config)
        connector.run()

