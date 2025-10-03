"""
IESO (Independent Electricity System Operator) Connector

Ingests Ontario electricity market data including HOEP (Hourly Ontario Energy Price),
pre-dispatch forecasts, and intertie flows.
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


class IESOConnector(Ingestor):
    """IESO Ontario electricity market data connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.ieso.ca/api")
        self.data_type = config.get("data_type", "HOEP")  # HOEP, PREDISPATCH, INTERTIE
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover IESO data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "hoep",
                    "market": "power",
                    "product": "energy",
                    "description": "Hourly Ontario Energy Price",
                    "update_freq": "hourly",
                    "currency": "CAD",
                },
                {
                    "name": "predispatch",
                    "market": "power",
                    "product": "energy",
                    "description": "Pre-Dispatch Price Forecast (3-hour rolling)",
                    "update_freq": "hourly",
                },
                {
                    "name": "intertie",
                    "market": "power",
                    "product": "transmission",
                    "description": "Intertie flows with MISO, NYISO, PJM, Manitoba, Quebec",
                    "update_freq": "5min",
                },
                {
                    "name": "ontario_demand",
                    "market": "power",
                    "product": "load",
                    "description": "Ontario Demand (OD)",
                    "update_freq": "5min",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull data from IESO API."""
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching IESO {self.data_type} data since {last_time}")
        
        if self.data_type == "HOEP":
            yield from self._fetch_hoep()
        elif self.data_type == "PREDISPATCH":
            yield from self._fetch_predispatch()
        elif self.data_type == "INTERTIE":
            yield from self._fetch_intertie()
        elif self.data_type == "DEMAND":
            yield from self._fetch_ontario_demand()
    
    def _fetch_hoep(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch HOEP (Hourly Ontario Energy Price).
        
        HOEP is the weighted average of Market Clearing Prices (MCP)
        for all dispatched hours in each hour.
        """
        now = datetime.utcnow()
        
        # Generate HOEP for each hour
        for hour in range(24):
            hour_ending = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Mock HOEP (typically CAD $20-100/MWh)
            base_price = 45.0
            
            # Ontario load shape (peak around 2-6 PM)
            if 14 <= hour <= 18:
                hourly_factor = 1.4  # Peak hours
            elif 7 <= hour <= 13 or 19 <= hour <= 22:
                hourly_factor = 1.2  # Shoulder hours
            else:
                hourly_factor = 0.8  # Off-peak hours
            
            # Add some randomness
            hoep = base_price * hourly_factor + (hash(str(hour)) % 15) - 7
            
            # HOEP components (mock)
            energy_component = hoep * 0.90
            global_adjustment = hoep * 0.10  # GA typically applied
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "IESO",
                "price_type": "HOEP",
                "price": hoep,
                "energy_component": energy_component,
                "global_adjustment": global_adjustment,
                "currency": "CAD",
                "hour_ending": hour_ending.isoformat(),
            }
    
    def _fetch_predispatch(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Pre-Dispatch prices.
        
        IESO publishes pre-dispatch schedules and prices every hour
        for the next 3 hours.
        """
        now = datetime.utcnow()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        
        # Forecast next 3 hours
        for hour_offset in range(1, 4):
            forecast_hour = current_hour + timedelta(hours=hour_offset)
            
            # Mock pre-dispatch price
            base_price = 48.0
            hour = forecast_hour.hour
            
            if 14 <= hour <= 18:
                price_forecast = base_price * 1.35
            elif 7 <= hour <= 13:
                price_forecast = base_price * 1.15
            else:
                price_forecast = base_price * 0.85
            
            # Add forecast uncertainty
            price_forecast += (hash(str(hour_offset)) % 10) - 5
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "IESO",
                "price_type": "PREDISPATCH",
                "forecast_price": price_forecast,
                "forecast_hour": forecast_hour.isoformat(),
                "currency": "CAD",
            }
    
    def _fetch_intertie(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch intertie flows.
        
        Ontario has interties with:
        - MISO (Michigan)
        - NYISO (New York)
        - PJM (via NYISO)
        - Manitoba Hydro
        - Hydro-Quebec
        """
        interties = [
            {
                "name": "IESO-MISO",
                "capacity_import": 1950,
                "capacity_export": 1950,
                "typical_flow": 500,
            },
            {
                "name": "IESO-NYISO",
                "capacity_import": 2250,
                "capacity_export": 1950,
                "typical_flow": 300,
            },
            {
                "name": "IESO-PJM",
                "capacity_import": 660,
                "capacity_export": 660,
                "typical_flow": 200,
            },
            {
                "name": "IESO-MANITOBA",
                "capacity_import": 300,
                "capacity_export": 300,
                "typical_flow": 150,
            },
            {
                "name": "IESO-QUEBEC",
                "capacity_import": 2410,
                "capacity_export": 1570,
                "typical_flow": 800,
            },
        ]
        
        now = datetime.utcnow()
        
        for intertie in interties:
            # Mock flow (MW)
            # Positive = import to Ontario, Negative = export from Ontario
            base_flow = intertie["typical_flow"]
            flow_variation = (hash(intertie["name"] + str(now.minute)) % 200) - 100
            flow = base_flow + flow_variation
            
            # Determine direction
            if flow > 0:
                direction = "IMPORT"
                scheduled_flow = min(flow, intertie["capacity_import"])
            else:
                direction = "EXPORT"
                scheduled_flow = max(flow, -intertie["capacity_export"])
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "IESO",
                "intertie": intertie["name"],
                "scheduled_flow_mw": scheduled_flow,
                "actual_flow_mw": scheduled_flow + (hash(str(now.second)) % 20) - 10,
                "direction": direction,
                "capacity_import_mw": intertie["capacity_import"],
                "capacity_export_mw": intertie["capacity_export"],
            }
    
    def _fetch_ontario_demand(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Ontario Demand (OD).
        
        Real-time electricity demand across Ontario.
        """
        now = datetime.utcnow()
        
        # Mock Ontario demand (typically 12,000-24,000 MW)
        base_demand = 16000  # MW
        
        # Hourly pattern
        hour = now.hour
        if 14 <= hour <= 18:
            hourly_factor = 1.4  # Peak
        elif 7 <= hour <= 13 or 19 <= hour <= 22:
            hourly_factor = 1.2  # Shoulder
        else:
            hourly_factor = 0.85  # Off-peak
        
        # Seasonal adjustment
        month = now.month
        if month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.3  # AC load
        elif month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.2  # Heating load
        else:
            seasonal_factor = 1.0
        
        ontario_demand = base_demand * hourly_factor * seasonal_factor
        ontario_demand += (hash(str(now.minute)) % 500) - 250
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "IESO",
            "metric": "ONTARIO_DEMAND",
            "demand_mw": ontario_demand,
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map IESO format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product, location, and value
        if "price_type" in raw and raw["price_type"] == "HOEP":
            product = "energy"
            location = "IESO.ONTARIO"
            value = raw["price"]
        elif "price_type" in raw and raw["price_type"] == "PREDISPATCH":
            product = "energy_forecast"
            location = "IESO.ONTARIO"
            value = raw["forecast_price"]
        elif "intertie" in raw:
            product = "transmission"
            location = raw["intertie"]
            value = raw["scheduled_flow_mw"]
        elif "demand_mw" in raw:
            product = "load"
            location = "IESO.ONTARIO"
            value = raw["demand_mw"]
        else:
            product = "unknown"
            location = "IESO.UNKNOWN"
            value = 0
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "trade",
            "value": float(value),
            "volume": None,
            "currency": raw.get("currency", "CAD"),
            "unit": "MWh" if "price" in str(product) else "MW",
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
        logger.info(f"Emitted {count} IESO events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"IESO checkpoint saved: {state}")


if __name__ == "__main__":
    # Test IESO connector
    configs = [
        {
            "source_id": "ieso_hoep",
            "data_type": "HOEP",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "ieso_predispatch",
            "data_type": "PREDISPATCH",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "ieso_intertie",
            "data_type": "INTERTIE",
            "kafka_topic": "power.transmission.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = IESOConnector(config)
        connector.run()

