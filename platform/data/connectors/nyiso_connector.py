"""
NYISO (New York Independent System Operator) Connector

Ingests Real-Time and Day-Ahead LBMP (Location-Based Marginal Price),
Capacity Market (ICAP), and Ancillary Services data from NYISO.
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


class NYISOConnector(Ingestor):
    """NYISO market data connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.nyiso.com/api")
        self.data_type = config.get("data_type", "RT_LBMP")  # RT_LBMP, DA_LBMP, ICAP, AS
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover NYISO data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "rt_lbmp",
                    "market": "power",
                    "product": "lbmp",
                    "update_freq": "5min",
                    "zones": "11 zones",
                },
                {
                    "name": "da_lbmp",
                    "market": "power",
                    "product": "lbmp",
                    "update_freq": "1day",
                    "zones": "11 zones",
                },
                {
                    "name": "icap",
                    "market": "power",
                    "product": "capacity",
                    "description": "Installed Capacity market",
                },
                {
                    "name": "ancillary_services",
                    "market": "power",
                    "product": "ancillary",
                    "description": "Regulation, Reserves, Voltage Support",
                },
                {
                    "name": "tcc",
                    "market": "power",
                    "product": "transmission",
                    "description": "Transmission Congestion Contracts",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull data from NYISO API."""
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching NYISO {self.data_type} data since {last_time}")
        
        if self.data_type == "RT_LBMP":
            yield from self._fetch_realtime_lbmp()
        elif self.data_type == "DA_LBMP":
            yield from self._fetch_dayahead_lbmp()
        elif self.data_type == "ICAP":
            yield from self._fetch_capacity_market()
        elif self.data_type == "AS":
            yield from self._fetch_ancillary_services()
        elif self.data_type == "TCC":
            yield from self._fetch_tcc_prices()
    
    def _fetch_realtime_lbmp(self) -> Iterator[Dict[str, Any]]:
        """Fetch Real-Time LBMP (5-minute intervals)."""
        # NYISO has 11 zones + interfaces
        zones = [
            "NYISO.ZONE.A",  # West
            "NYISO.ZONE.B",  # Genesee
            "NYISO.ZONE.C",  # Central
            "NYISO.ZONE.D",  # North
            "NYISO.ZONE.E",  # Mohawk Valley
            "NYISO.ZONE.F",  # Capital
            "NYISO.ZONE.G",  # Hudson Valley
            "NYISO.ZONE.H",  # Millwood
            "NYISO.ZONE.I",  # Dunwoodie
            "NYISO.ZONE.J",  # NYC
            "NYISO.ZONE.K",  # Long Island
        ]
        
        # Current 5-min interval
        now = datetime.utcnow()
        interval_ending = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        
        for zone in zones:
            # Mock LBMP with components
            base_price = 40.0
            
            # NYC and Long Island typically higher
            if "J" in zone or "K" in zone:
                base_price = 50.0
            
            energy = base_price + (hash(zone) % 15) - 7
            congestion = max(-10, min(15, (hash(zone + "cong") % 20) - 10))
            loss = max(0, (hash(zone + "loss") % 4) - 1)
            
            lbmp = energy + congestion + loss
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "zone": zone,
                "lbmp": lbmp,
                "energy_component": energy,
                "congestion_component": congestion,
                "loss_component": loss,
                "interval_ending": interval_ending.isoformat(),
                "market": "RTBM",
            }
    
    def _fetch_dayahead_lbmp(self) -> Iterator[Dict[str, Any]]:
        """Fetch Day-Ahead LBMP."""
        zones = [
            "NYISO.ZONE.J",  # NYC
            "NYISO.ZONE.K",  # Long Island
            "NYISO.ZONE.A",  # West
            "NYISO.ZONE.G",  # Hudson Valley
        ]
        
        # DA market for tomorrow
        tomorrow = datetime.utcnow() + timedelta(days=1)
        
        for hour in range(24):
            interval = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            for zone in zones:
                # Mock DA LBMP
                base_price = 45.0 if "J" in zone or "K" in zone else 38.0
                
                # Load shape (peak in afternoon)
                hourly_factor = 1 + 0.4 * (abs(hour - 15) / 15)
                
                energy = base_price * hourly_factor
                congestion = (hash(zone + str(hour)) % 12) - 6
                loss = max(0, (hash(zone) % 3) - 1)
                
                lbmp = energy + congestion + loss
                
                yield {
                    "timestamp": datetime.utcnow().isoformat(),
                    "zone": zone,
                    "lbmp": lbmp,
                    "energy_component": energy,
                    "congestion_component": congestion,
                    "loss_component": loss,
                    "interval_ending": interval.isoformat(),
                    "market": "DAM",
                }
    
    def _fetch_capacity_market(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch ICAP (Installed Capacity) market prices.
        
        NYISO runs monthly capacity auctions.
        """
        # Capacity zones
        capacity_zones = [
            "NYISO.ICAP.G-J",  # NYC/LI
            "NYISO.ICAP.REST_OF_STATE",
        ]
        
        # Current month
        now = datetime.utcnow()
        capability_period = now.strftime("%Y-%m")
        
        for zone in capacity_zones:
            # Mock ICAP clearing price ($/kW-month)
            if "G-J" in zone:
                clearing_price = 8.50 + (hash(zone) % 3)  # $8-11/kW-mo for NYC/LI
            else:
                clearing_price = 2.00 + (hash(zone) % 2)  # $2-4/kW-mo for ROS
            
            # Mock cleared MW
            cleared_mw = 5000 + (hash(zone + str(now.month)) % 2000)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "zone": zone,
                "clearing_price": clearing_price,
                "cleared_mw": cleared_mw,
                "capability_period": capability_period,
                "market": "ICAP",
                "unit": "$/kW-month",
            }
    
    def _fetch_ancillary_services(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Ancillary Services prices.
        
        Includes Regulation, 10-min Spinning Reserve, 10-min Non-Sync Reserve,
        and 30-min Operating Reserve.
        """
        now = datetime.utcnow()
        interval_ending = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        
        # AS products
        as_products = [
            {"type": "REGULATION", "typical_price": 15.0},
            {"type": "SPIN_10MIN", "typical_price": 12.0},
            {"type": "NON_SYNC_10MIN", "typical_price": 8.0},
            {"type": "OR_30MIN", "typical_price": 5.0},
        ]
        
        for product in as_products:
            # Mock AS price
            price = product["typical_price"] + (hash(product["type"]) % 6) - 3
            
            # Mock MW procured
            mw_procured = 300 + (hash(product["type"] + str(now.hour)) % 200)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "product": product["type"],
                "price": max(0, price),
                "mw_procured": mw_procured,
                "interval_ending": interval_ending.isoformat(),
                "market": "AS",
            }
    
    def _fetch_tcc_prices(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Transmission Congestion Contract prices.
        
        TCCs provide congestion hedging between zones.
        """
        # Sample TCC paths
        tcc_paths = [
            {"from": "NYISO.ZONE.A", "to": "NYISO.ZONE.J", "name": "WEST_TO_NYC"},
            {"from": "NYISO.ZONE.G", "to": "NYISO.ZONE.J", "name": "HV_TO_NYC"},
            {"from": "NYISO.ZONE.J", "to": "NYISO.ZONE.K", "name": "NYC_TO_LI"},
        ]
        
        now = datetime.utcnow()
        
        for path in tcc_paths:
            # Mock TCC clearing price ($/MW)
            # Based on expected congestion
            if "NYC" in path["to"]:
                clearing_price = 5.0 + (hash(path["name"]) % 8)
            else:
                clearing_price = 1.0 + (hash(path["name"]) % 3)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "tcc_name": path["name"],
                "from_zone": path["from"],
                "to_zone": path["to"],
                "clearing_price": clearing_price,
                "market": "TCC",
                "unit": "$/MW",
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map NYISO format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product type and value
        if "lbmp" in raw:
            product = "lbmp"
            value = raw["lbmp"]
            location = raw["zone"]
        elif "clearing_price" in raw and "ICAP" in raw.get("market", ""):
            product = "capacity"
            value = raw["clearing_price"]
            location = raw["zone"]
        elif "product" in raw and raw.get("market") == "AS":
            product = "ancillary"
            value = raw["price"]
            location = f"NYISO.AS.{raw['product']}"
        elif "tcc_name" in raw:
            product = "transmission"
            value = raw["clearing_price"]
            location = f"NYISO.TCC.{raw['tcc_name']}"
        else:
            product = "unknown"
            value = 0
            location = "NYISO.UNKNOWN"
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "trade",
            "value": float(value),
            "volume": raw.get("mw_procured") or raw.get("cleared_mw"),
            "currency": "USD",
            "unit": raw.get("unit", "MWh"),
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
        logger.info(f"Emitted {count} NYISO events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"NYISO checkpoint saved: {state}")


if __name__ == "__main__":
    # Test NYISO connector
    configs = [
        {
            "source_id": "nyiso_rtbm",
            "data_type": "RT_LBMP",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "nyiso_dam",
            "data_type": "DA_LBMP",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "nyiso_icap",
            "data_type": "ICAP",
            "kafka_topic": "power.capacity.v1",
        },
        {
            "source_id": "nyiso_as",
            "data_type": "AS",
            "kafka_topic": "power.ancillary.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = NYISOConnector(config)
        connector.run()

