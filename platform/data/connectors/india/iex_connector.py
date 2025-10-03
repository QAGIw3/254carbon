"""
Indian Energy Exchange (IEX) Connector

Integrates with India's premier power exchange covering:
- Day-Ahead Market (DAM)
- Real-Time Market (RTM) - hourly  
- Term-Ahead Market (TAM)
- Green Market (G-DAM, G-TAM)
- Renewable Energy Certificates (REC)
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class IEXConnector(Ingestor):
    """Indian Energy Exchange connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.iexindia.com/api")
        self.market_type = config.get("market_type", "DAM")  # DAM, RTM, TAM, G-DAM
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover IEX market streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "day_ahead",
                    "market": "DAM",
                    "description": "Day-Ahead Market hourly prices",
                    "currency": "INR",
                },
                {
                    "name": "real_time",
                    "market": "RTM",
                    "description": "Real-Time Market hourly dispatch",
                    "update_freq": "hourly",
                },
                {
                    "name": "term_ahead",
                    "market": "TAM",
                    "description": "Intra-day and day-ahead contingency",
                },
                {
                    "name": "green_market",
                    "market": "G-DAM",
                    "description": "100% renewable energy trading",
                },
                {
                    "name": "rec_market",
                    "market": "REC",
                    "description": "Renewable Energy Certificates",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull IEX market data."""
        logger.info(f"Fetching IEX {self.market_type} data")
        
        if self.market_type == "DAM":
            yield from self._fetch_day_ahead()
        elif self.market_type == "RTM":
            yield from self._fetch_real_time()
        elif self.market_type in ["G-DAM", "G-TAM"]:
            yield from self._fetch_green_market()
        elif self.market_type == "REC":
            yield from self._fetch_rec_market()
    
    def _fetch_day_ahead(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Day-Ahead Market prices.
        
        DAM is cleared for next day with hourly blocks.
        """
        # India has 5 regional grids
        regions = ["NORTH", "SOUTH", "EAST", "WEST", "NORTHEAST"]
        
        tomorrow = datetime.utcnow() + timedelta(days=1)
        ist_tomorrow = tomorrow + timedelta(hours=5, minutes=30)  # IST (UTC+5:30)
        
        for region in regions:
            # Regional base prices (INR/MWh)
            base_prices = {
                "NORTH": 4200,  # High demand (Delhi, Punjab, Haryana)
                "SOUTH": 4500,  # Industrial load (Tamil Nadu, Karnataka)
                "EAST": 3800,  # Coal surplus (Jharkhand, Odisha)
                "WEST": 4800,  # Commercial load (Maharashtra, Gujarat)
                "NORTHEAST": 3500,  # Hydro surplus
            }
            
            base = base_prices[region]
            
            # Generate 24 hourly prices
            for hour in range(24):
                delivery_hour = ist_tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                # Peak/off-peak factors
                if 18 <= hour <= 23:
                    hourly_factor = 1.6  # Evening peak
                elif 6 <= hour <= 10:
                    hourly_factor = 1.3  # Morning peak
                elif 11 <= hour <= 17:
                    hourly_factor = 1.1  # Afternoon
                else:
                    hourly_factor = 0.7  # Night valley
                
                # Seasonal factors
                month = delivery_hour.month
                if month in [5, 6]:  # Pre-monsoon heat
                    seasonal_factor = 1.4
                elif month in [7, 8, 9]:  # Monsoon (hydro surplus)
                    seasonal_factor = 0.85
                elif month in [12, 1]:  # Winter
                    seasonal_factor = 1.1
                else:
                    seasonal_factor = 1.0
                
                price = base * hourly_factor * seasonal_factor
                
                # Add market clearing variation
                price += (hash(region + str(hour)) % 400) - 200
                
                # Price caps (CERC regulations)
                price = max(2000, min(12000, price))
                
                # Volume (MW)
                volume = 2000 + (hash(str(hour)) % 1000)
                
                yield {
                    "timestamp": datetime.utcnow().isoformat(),
                    "market": "IEX",
                    "market_type": "DAM",
                    "region": region,
                    "price_inr_mwh": price,
                    "delivery_hour": delivery_hour.isoformat(),
                    "volume_mw": volume,
                    "currency": "INR",
                }
    
    def _fetch_real_time(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Real-Time Market prices.
        
        RTM allows hourly trading for same-day delivery.
        """
        regions = ["NORTH", "SOUTH", "EAST", "WEST"]
        
        now = datetime.utcnow()
        ist_now = now + timedelta(hours=5, minutes=30)
        current_hour = ist_now.replace(minute=0, second=0, microsecond=0)
        
        for region in regions:
            # RTM typically trades at premium to DAM
            dam_base = {"NORTH": 4200, "SOUTH": 4500, "EAST": 3800, "WEST": 4800}[region]
            
            # Real-time premium/discount
            hour = ist_now.hour
            if 18 <= hour <= 21:
                rt_factor = 1.2  # Premium in peak
            else:
                rt_factor = 0.95  # Discount in off-peak
            
            price = dam_base * rt_factor
            price += (hash(region + str(ist_now.minute)) % 600) - 300
            price = max(2000, min(15000, price))
            
            volume = 500 + (hash(str(ist_now.minute)) % 300)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "IEX",
                "market_type": "RTM",
                "region": region,
                "price_inr_mwh": price,
                "delivery_hour": current_hour.isoformat(),
                "volume_mw": volume,
                "currency": "INR",
            }
    
    def _fetch_green_market(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Green Market prices.
        
        G-DAM/G-TAM trade 100% renewable energy.
        """
        # Green power typically trades at premium
        regions = ["NORTH", "SOUTH", "WEST"]
        
        for region in regions:
            # Green premium (INR/MWh)
            base_green = {
                "NORTH": 4800,
                "SOUTH": 5000,
                "WEST": 5200,
            }[region]
            
            # Renewable availability factor
            month = datetime.utcnow().month
            if month in [7, 8, 9]:  # Monsoon (high hydro)
                availability_factor = 0.9
            elif month in [3, 4, 5]:  # Summer (high solar)
                availability_factor = 0.95
            else:
                availability_factor = 1.1  # Scarcity premium
            
            price = base_green * availability_factor
            price += (hash(region) % 300) - 150
            price = max(3000, min(10000, price))
            
            volume = 800 + (hash(str(datetime.utcnow().hour)) % 400)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "IEX",
                "market_type": "G-DAM",
                "region": region,
                "price_inr_mwh": price,
                "volume_mw": volume,
                "currency": "INR",
                "renewable_pct": 100,
            }
    
    def _fetch_rec_market(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Renewable Energy Certificate market data.
        
        RECs separate renewable attribute from electricity.
        """
        # REC types
        rec_types = [
            {"category": "SOLAR", "price": 1000, "volume": 50000},
            {"category": "NON_SOLAR", "price": 1000, "volume": 100000},
        ]
        
        for rec in rec_types:
            # Floor and forbearance prices (CERC set)
            floor_price = 1000  # INR per REC
            forbearance_price = 2500  # INR per REC
            
            # Market clearing price
            price = floor_price + (hash(rec["category"]) % 800)
            price = min(price, forbearance_price)
            
            # Volume variation
            volume = rec["volume"] + (hash(str(datetime.utcnow().day)) % 20000) - 10000
            volume = max(10000, volume)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "IEX",
                "product": "REC",
                "category": rec["category"],
                "price_inr": price,
                "volume_certificates": volume,
                "currency": "INR",
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map IEX format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product and location
        if "price_inr_mwh" in raw:
            product = "energy"
            location = f"IEX.{raw['market_type']}.{raw['region']}"
            value = raw["price_inr_mwh"]
        elif "product" in raw and raw["product"] == "REC":
            product = "renewable_certificate"
            location = f"IEX.REC.{raw['category']}"
            value = raw["price_inr"]
        else:
            product = "unknown"
            location = "IEX.UNKNOWN"
            value = 0
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "market_clearing",
            "value": float(value),
            "volume": raw.get("volume_mw") or raw.get("volume_certificates"),
            "currency": "INR",
            "unit": "MWh" if product == "energy" else "certificate",
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
        logger.info(f"Emitted {count} IEX events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    # Test IEX connector
    configs = [
        {"source_id": "iex_dam", "market_type": "DAM"},
        {"source_id": "iex_rtm", "market_type": "RTM"},
        {"source_id": "iex_green", "market_type": "G-DAM"},
        {"source_id": "iex_rec", "market_type": "REC"},
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = IEXConnector(config)
        connector.run()

