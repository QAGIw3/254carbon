"""
ERCOT Connector

Ingests Settlement Point Prices (SPP), hub prices, ORDC adders,
and resource-specific data from ERCOT.
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


class ERCOTConnector(Ingestor):
    """
    ERCOT market data connector.

    Responsibilities
    - Discover SPP, hub prices, ORDC adders, and resource telemetry streams
    - Pull data (mocked in this scaffold) and map to canonical schema
    - Emit to Kafka with validation and sequencing

    Production notes
    - API base: https://api.ercot.com/api/public-reports
    - Real integrations typically require dataset-specific paths/filters
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://api.ercot.com/api/public-reports")
        self.api_subscription_key = config.get("api_subscription_key")
        self.data_type = config.get("data_type", "SPP")  # SPP, HUB, ORDC, RESOURCE
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover ERCOT data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "spp_realtime",
                    "description": "Settlement Point Prices (SPP) - Real-time",
                    "frequency": "5min",
                    "data_points": "~1000 nodes",
                    "endpoint": "/np4-181-cd/spp_realtime"
                },
                {
                    "name": "spp_day_ahead",
                    "description": "Settlement Point Prices (SPP) - Day-ahead",
                    "frequency": "hourly",
                    "data_points": "~1000 nodes",
                    "endpoint": "/np4-181-cd/spp_day_ahead"
                },
                {
                    "name": "hub_prices",
                    "description": "Hub Prices (HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST)",
                    "frequency": "5min",
                    "data_points": "4 hubs",
                    "endpoint": "/np4-181-cd/hub_prices"
                },
                {
                    "name": "ordc_adders",
                    "description": "Operating Reserve Demand Curve (ORDC) Adders",
                    "frequency": "5min",
                    "data_points": "system-wide",
                    "endpoint": "/np4-181-cd/ordc_adders"
                },
                {
                    "name": "resource_telemetry",
                    "description": "Resource-specific telemetry data",
                    "frequency": "5min",
                    "data_points": "~500 resources",
                    "endpoint": "/np4-181-cd/resource_telemetry"
                }
            ]
        }
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "15min",  # ERCOT uses 15-min intervals
                    "nodes": "~4000",  # ERCOT has ~4000 settlement points
                },
                {
                    "name": "hub_prices",
                    "market": "power",
                    "product": "lmp",
                    "hubs": ["NORTH", "SOUTH", "WEST", "HOUSTON"],
                },
                {
                    "name": "ordc_adders",
                    "market": "power",
                    "product": "ancillary",
                    "description": "Operating Reserve Demand Curve adders",
                },
                {
                    "name": "resource_telemetry",
                    "market": "power",
                    "product": "generation",
                    "description": "Resource-specific generation and availability",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull data from ERCOT API.
        
        ERCOT provides data through their Public API.
        """
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching ERCOT {self.data_type} data since {last_time}")
        
        if self.data_type == "SPP":
            yield from self._fetch_spp()
        elif self.data_type == "HUB":
            yield from self._fetch_hub_prices()
        elif self.data_type == "ORDC":
            yield from self._fetch_ordc_adders()
        elif self.data_type == "RESOURCE":
            yield from self._fetch_resource_data()
    
    def _fetch_spp(self) -> Iterator[Dict[str, Any]]:
        """Fetch Settlement Point Prices (15-minute intervals, mock)."""
        # Mock data - in production would call ERCOT API
        # Example: GET /np6-345-cd/spp_node_zone_hub
        
        # Sample ERCOT settlement points
        settlement_points = [
            "ERCOT.SP.LZ_NORTH",
            "ERCOT.SP.LZ_SOUTH",
            "ERCOT.SP.LZ_WEST",
            "ERCOT.SP.LZ_HOUSTON",
            "ERCOT.SP.LZ_EAST",
            "ERCOT.SP.LZ_WEST",
            "ERCOT.SP.LZ_LCRA",
            "ERCOT.SP.LZ_RAYBN",
        ]
        
        # Current 15-min interval
        now = datetime.utcnow()
        interval_ending = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        
        for sp in settlement_points:
            # Mock SPP
            base_price = 40.0
            spp = base_price + (hash(sp) % 30) - 15  # ±15 variation
            
            # ORDC adder (scarcity pricing)
            ordc_adder = max(0, (hash(sp + "ordc") % 50) - 40)  # Can be 0-10 $/MWh
            
            total_spp = spp + ordc_adder
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "settlement_point": sp,
                "spp": total_spp,
                "energy_price": spp,
                "ordc_adder": ordc_adder,
                "interval_ending": interval_ending.isoformat(),
                "market": "RTM",  # Real-Time Market
            }
    
    def _fetch_hub_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch hub prices for major load zones (mock)."""
        hubs = {
            "ERCOT.HUB.NORTH": 42.0,
            "ERCOT.HUB.SOUTH": 44.0,
            "ERCOT.HUB.WEST": 39.0,
            "ERCOT.HUB.HOUSTON": 45.0,
        }
        
        now = datetime.utcnow()
        interval_ending = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        
        for hub_id, base_price in hubs.items():
            # Add some variation
            price = base_price + (hash(hub_id + str(now.hour)) % 10) - 5
            
            # ORDC component
            ordc = max(0, (hash(hub_id + "ordc") % 30) - 25)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "hub_id": hub_id,
                "price": price + ordc,
                "energy_component": price,
                "ordc_component": ordc,
                "interval_ending": interval_ending.isoformat(),
                "market": "RTM",
            }
    
    def _fetch_ordc_adders(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Operating Reserve Demand Curve (ORDC) adders (mock).

        ORDC adds scarcity pricing when reserves are tight.
        """
        now = datetime.utcnow()
        interval_ending = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        
        # System-wide ORDC
        reserves_online = 2500  # MW (mock)
        reserves_offline = 1800  # MW (mock)
        
        # ORDC calculation (simplified)
        # Real ORDC uses complex curve based on LOLP
        if reserves_online < 2000:
            ordc_online = (2000 - reserves_online) * 0.5  # $/MWh
        else:
            ordc_online = 0.0
        
        if reserves_offline < 1500:
            ordc_offline = (1500 - reserves_offline) * 0.3
        else:
            ordc_offline = 0.0
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "interval_ending": interval_ending.isoformat(),
            "ordc_online": ordc_online,
            "ordc_offline": ordc_offline,
            "total_ordc": ordc_online + ordc_offline,
            "reserves_online_mw": reserves_online,
            "reserves_offline_mw": reserves_offline,
        }
    
    def _fetch_resource_data(self) -> Iterator[Dict[str, Any]]:
        """Fetch resource-specific generation and telemetry (mock)."""
        # Sample resources
        resources = [
            {"id": "ERCOT.GEN.WIND_01", "type": "WIND", "capacity": 250},
            {"id": "ERCOT.GEN.SOLAR_01", "type": "SOLAR", "capacity": 150},
            {"id": "ERCOT.GEN.GAS_01", "type": "GAS", "capacity": 500},
        ]
        
        for resource in resources:
            # Mock generation (capacity factor)
            if resource["type"] == "WIND":
                cf = 0.30 + (hash(resource["id"]) % 30) / 100
            elif resource["type"] == "SOLAR":
                hour = datetime.utcnow().hour
                cf = max(0, 0.8 * (1 - abs(hour - 12) / 12)) if 6 <= hour <= 18 else 0
            else:
                cf = 0.85  # Gas plant
            
            generation = resource["capacity"] * cf
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "resource_id": resource["id"],
                "resource_type": resource["type"],
                "capacity_mw": resource["capacity"],
                "generation_mw": generation,
                "capacity_factor": cf,
                "status": "ONLINE",
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map ERCOT format to canonical schema for SPP/Hub/ORDC/Generation."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product type
        if "spp" in raw:
            product = "lmp"
            value = raw["spp"]
            location = raw["settlement_point"]
        elif "hub_id" in raw:
            product = "lmp"
            value = raw["price"]
            location = raw["hub_id"]
        elif "ordc_online" in raw:
            product = "ordc"
            value = raw["total_ordc"]
            location = "ERCOT.SYSTEM"
        else:
            product = "generation"
            value = raw.get("generation_mw", 0)
            location = raw.get("resource_id", "ERCOT.UNKNOWN")
        
        payload = {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "trade",
            "value": float(value),
            "volume": None,
            "currency": "USD",
            "unit": "MWh" if product in ["lmp", "ordc"] else "MW",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }
        # Promote components if present in raw
        if raw.get("energy_component") is not None:
            payload["energy_component"] = float(raw["energy_component"])  # type: ignore[arg-type]
        if raw.get("congestion_component") is not None:
            payload["congestion_component"] = float(raw["congestion_component"])  # type: ignore[arg-type]
        if raw.get("loss_component") is not None:
            payload["loss_component"] = float(raw["loss_component"])  # type: ignore[arg-type]
        # ORDC components can be preserved in metadata or promoted separately if schema supports
        if raw.get("ordc_component") is not None:
            payload["ordc_component"] = float(raw["ordc_component"])  # type: ignore[arg-type]
        return payload
    
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
        logger.info(f"Emitted {count} ERCOT events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"ERCOT checkpoint saved: {state}")


if __name__ == "__main__":
    # Test ERCOT connector
    configs = [
        {
            "source_id": "ercot_spp",
            "data_type": "SPP",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "ercot_hubs",
            "data_type": "HUB",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "ercot_ordc",
            "data_type": "ORDC",
            "kafka_topic": "power.ordc.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = ERCOTConnector(config)
        connector.run()
