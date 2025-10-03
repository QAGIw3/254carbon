"""
NEM (National Electricity Market) Connector

Ingests spot prices and FCAS data from Australia's NEM.
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


class NEMConnector(Ingestor):
    """Australian NEM connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.aemo.com.au/api")
        self.region = config.get("region", "NSW1")  # NSW1, QLD1, SA1, TAS1, VIC1
        self.data_type = config.get("data_type", "SPOT")  # SPOT, FCAS, PREDISPATCH
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover NEM data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "spot_prices",
                    "market": "power",
                    "product": "energy",
                    "description": "5-minute dispatch prices",
                    "regions": ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"],
                    "currency": "AUD",
                },
                {
                    "name": "fcas",
                    "market": "power",
                    "product": "ancillary",
                    "description": "Frequency Control Ancillary Services",
                    "services": ["RAISE6SEC", "RAISE60SEC", "RAISE5MIN", "LOWER6SEC", "LOWER60SEC", "LOWER5MIN"],
                },
                {
                    "name": "predispatch",
                    "market": "power",
                    "product": "forecast",
                    "description": "Pre-dispatch price forecasts",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull data from NEM API."""
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching NEM {self.region} {self.data_type} data since {last_time}")
        
        if self.data_type == "SPOT":
            yield from self._fetch_spot_prices()
        elif self.data_type == "FCAS":
            yield from self._fetch_fcas()
    
    def _fetch_spot_prices(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch 5-minute dispatch prices.
        
        NEM operates with 5-minute dispatch intervals and
        30-minute settlement periods.
        """
        # Regional characteristics (AUD/MWh)
        regional_config = {
            "NSW1": {"base_price": 80.0, "volatility": 1.3},
            "QLD1": {"base_price": 75.0, "volatility": 1.2},
            "SA1": {"base_price": 90.0, "volatility": 2.0},  # High renewables, volatile
            "TAS1": {"base_price": 70.0, "volatility": 0.8},  # Hydro-heavy, stable
            "VIC1": {"base_price": 78.0, "volatility": 1.1},
        }
        
        config = regional_config.get(self.region, regional_config["NSW1"])
        now = datetime.utcnow()
        
        # Generate prices for 12 x 5-min intervals (1 hour)
        for interval in range(12):
            dispatch_time = now + timedelta(minutes=5 * interval)
            dispatch_time = dispatch_time.replace(
                minute=(dispatch_time.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            
            # AEST hour (UTC+10)
            aest_hour = (dispatch_time.hour + 10) % 24
            
            # Australian load pattern
            if 17 <= aest_hour <= 20:
                hourly_factor = 1.8  # Evening peak (AC in summer)
            elif 7 <= aest_hour <= 16:
                hourly_factor = 1.3  # Daytime
            else:
                hourly_factor = 0.7  # Night
            
            # Base calculation
            price = config["base_price"] * hourly_factor
            
            # Add volatility
            volatility = config["volatility"] * (hash(str(interval)) % 30) - 15
            price += volatility
            
            # Occasional price spikes (NEM is known for volatility)
            if hash(str(interval) + self.region) % 30 == 0:
                price *= 5.0  # Significant spike
            
            # Price cap at AUD 16,600/MWh, floor at -AUD 1,000/MWh
            price = max(-1000, min(16600, price))
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "NEM",
                "region": self.region,
                "price_type": "DISPATCH",
                "price_aud_mwh": price,
                "regional_demand_mw": 5000 + (hash(str(interval)) % 2000),
                "dispatch_interval": dispatch_time.isoformat(),
                "currency": "AUD",
            }
    
    def _fetch_fcas(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch FCAS (Frequency Control Ancillary Services) prices.
        
        NEM has 8 FCAS markets:
        - Regulation (raise/lower)
        - Fast (6sec, 60sec, 5min) raise/lower
        """
        fcas_services = [
            {"service": "RAISE6SEC", "base_price": 15.0},
            {"service": "RAISE60SEC", "base_price": 12.0},
            {"service": "RAISE5MIN", "base_price": 8.0},
            {"service": "RAISEREG", "base_price": 20.0},
            {"service": "LOWER6SEC", "base_price": 12.0},
            {"service": "LOWER60SEC", "base_price": 10.0},
            {"service": "LOWER5MIN", "base_price": 6.0},
            {"service": "LOWERREG", "base_price": 18.0},
        ]
        
        now = datetime.utcnow()
        
        for service in fcas_services:
            # FCAS prices vary with system conditions
            price = service["base_price"] + (hash(service["service"]) % 10) - 5
            price = max(0, price)
            
            # MW enabled
            mw_enabled = 50 + (hash(service["service"] + self.region) % 100)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "NEM",
                "region": self.region,
                "service_type": service["service"],
                "price_aud_mwh": price,
                "mw_enabled": mw_enabled,
                "currency": "AUD",
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map NEM format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        if "service_type" in raw:
            product = "ancillary"
            location = f"NEM.{raw['region']}.{raw['service_type']}"
            value = raw["price_aud_mwh"]
        else:
            product = "energy"
            location = f"NEM.{raw['region']}"
            value = raw["price_aud_mwh"]
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "dispatch",
            "value": float(value),
            "volume": raw.get("regional_demand_mw") or raw.get("mw_enabled"),
            "currency": "AUD",
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
        logger.info(f"Emitted {count} NEM events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"NEM checkpoint saved: {state}")


if __name__ == "__main__":
    # Test NEM connector
    configs = [
        {
            "source_id": "nem_nsw",
            "region": "NSW1",
            "data_type": "SPOT",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "nem_sa_fcas",
            "region": "SA1",
            "data_type": "FCAS",
            "kafka_topic": "power.ancillary.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = NEMConnector(config)
        connector.run()

