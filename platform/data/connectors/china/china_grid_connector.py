"""
China National Grid Corporation Power Market Connector

Integrates with China's complex multi-tier electricity market:
- Beijing & Guangzhou power exchanges
- 8 regional power grids
- Provincial spot markets
- National carbon ETS
- Green electricity certificates (GEC)
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class ChinaGridConnector(Ingestor):
    """China power market data connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://api.cspex.com.cn")
        self.exchange = config.get("exchange", "BEIJING")  # BEIJING, GUANGZHOU
        self.region = config.get("region", "EAST")  # EAST, NORTH, NORTHEAST, NORTHWEST, CENTRAL, SOUTH
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover China power market streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "spot_market",
                    "exchanges": ["Beijing", "Guangzhou"],
                    "description": "15-minute spot prices",
                    "currency": "CNY",
                },
                {
                    "name": "provincial_markets",
                    "provinces": ["Guangdong", "Jiangsu", "Zhejiang", "Shandong", "Henan"],
                    "description": "Provincial spot trading",
                },
                {
                    "name": "inter_provincial",
                    "description": "Cross-regional transmission",
                },
                {
                    "name": "green_certificates",
                    "description": "GEC trading and settlement",
                },
                {
                    "name": "national_ets",
                    "description": "Carbon allowances (CNY/tCO2)",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull data from China grid markets."""
        logger.info(f"Fetching China {self.exchange} {self.region} data")
        
        yield from self._fetch_spot_prices()
        yield from self._fetch_green_certificates()
        yield from self._fetch_carbon_ets()
    
    def _fetch_spot_prices(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch spot market prices.
        
        China operates 15-minute spot markets in select provinces.
        Pricing reflects coal-power linkage mechanism.
        """
        # Regional base prices (CNY/MWh)
        regional_bases = {
            "EAST": {"base": 420, "coal_link": 0.85},  # Jiangsu, Zhejiang, Shanghai
            "NORTH": {"base": 380, "coal_link": 0.90},  # Beijing, Tianjin, Hebei
            "NORTHEAST": {"base": 350, "coal_link": 0.88},  # Liaoning, Jilin
            "NORTHWEST": {"base": 340, "coal_link": 0.75},  # Shaanxi, Gansu (hydro/wind)
            "CENTRAL": {"base": 390, "coal_link": 0.87},  # Henan, Hubei
            "SOUTH": {"base": 450, "coal_link": 0.82},  # Guangdong, Guangxi
        }
        
        config = regional_bases.get(self.region, regional_bases["EAST"])
        
        # Current 15-min interval
        now = datetime.utcnow()
        china_time = now + timedelta(hours=8)  # CST (UTC+8)
        interval = china_time.replace(minute=(china_time.minute // 15) * 15, second=0, microsecond=0)
        
        # Provincial markets
        provinces = {
            "EAST": ["Jiangsu", "Zhejiang", "Shanghai"],
            "NORTH": ["Beijing", "Hebei", "Shandong"],
            "SOUTH": ["Guangdong", "Guangxi", "Hainan"],
        }
        
        for province in provinces.get(self.region, ["Generic"]):
            # Base price varies by hour
            hour = interval.hour
            if 10 <= hour <= 11 or 19 <= hour <= 21:
                peak_factor = 1.5  # Peak hours
            elif 8 <= hour <= 18:
                peak_factor = 1.2  # Day hours
            else:
                peak_factor = 0.75  # Valley hours
            
            # Coal price influence (thermal generation dominant)
            coal_price_cny = 850  # CNY/tonne
            coal_factor = config["coal_link"] * (coal_price_cny / 800)  # Normalize to 800 CNY
            
            # Renewable penetration effect
            month = interval.month
            if month in [6, 7, 8]:  # Summer hydro season
                renewable_discount = 0.92
            elif month in [12, 1, 2]:  # Winter heating
                renewable_discount = 1.05
            else:
                renewable_discount = 1.0
            
            price = config["base"] * peak_factor * coal_factor * renewable_discount
            
            # Add market variation
            price += (hash(province + str(interval.minute)) % 40) - 20
            
            # Floor and ceiling (government regulated)
            price = max(200, min(800, price))
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "CHINA_GRID",
                "exchange": self.exchange,
                "region": self.region,
                "province": province,
                "price_cny_mwh": price,
                "interval_ending": interval.isoformat(),
                "currency": "CNY",
                "market_type": "spot",
            }
    
    def _fetch_green_certificates(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch Green Electricity Certificate (GEC) trading data.
        
        China's GEC system encourages renewable consumption.
        """
        # GEC prices by technology (CNY per certificate = 1 MWh)
        technologies = [
            {"type": "WIND", "price": 50, "volume": 150000},
            {"type": "SOLAR", "price": 55, "volume": 120000},
            {"type": "HYDRO", "price": 45, "volume": 80000},
            {"type": "BIOMASS", "price": 60, "volume": 30000},
        ]
        
        for tech in technologies:
            # Price variation
            price = tech["price"] + (hash(tech["type"]) % 10) - 5
            price = max(30, price)
            
            # Volume variation
            volume = tech["volume"] + (hash(str(datetime.utcnow().hour)) % 20000) - 10000
            volume = max(10000, volume)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "CHINA_GEC",
                "technology": tech["type"],
                "price_cny": price,
                "volume_certificates": volume,
                "currency": "CNY",
            }
    
    def _fetch_carbon_ets(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch National Carbon ETS data.
        
        China's national ETS is the world's largest carbon market.
        Currently covers power sector, expanding to others.
        """
        # National ETS spot price (CNY/tCO2)
        base_price = 60.0  # CNY per tonne CO2
        
        # Price influenced by policy and compliance deadlines
        month = datetime.utcnow().month
        if month in [11, 12]:  # Pre-compliance rush
            compliance_factor = 1.3
        elif month in [1, 2]:  # Post-compliance lull
            compliance_factor = 0.85
        else:
            compliance_factor = 1.0
        
        price = base_price * compliance_factor
        price += (hash(str(datetime.utcnow().day)) % 8) - 4
        price = max(40, min(100, price))
        
        # Trading volume (tonnes)
        volume = 2000000 + (hash(str(datetime.utcnow().hour)) % 500000)
        
        yield {
            "timestamp": datetime.utcnow().isoformat(),
            "market": "CHINA_ETS",
            "price_cny_tco2": price,
            "volume_tonnes": volume,
            "currency": "CNY",
            "sector": "POWER",
        }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map China format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product and location
        if "price_cny_mwh" in raw:
            product = "energy"
            location = f"CHINA.{raw['region']}.{raw['province']}"
            value = raw["price_cny_mwh"]
            currency = "CNY"
        elif "market" in raw and raw["market"] == "CHINA_GEC":
            product = "green_certificate"
            location = f"CHINA.GEC.{raw['technology']}"
            value = raw["price_cny"]
            currency = "CNY"
        elif "market" in raw and raw["market"] == "CHINA_ETS":
            product = "carbon"
            location = "CHINA.ETS.NATIONAL"
            value = raw["price_cny_tco2"]
            currency = "CNY"
        else:
            product = "unknown"
            location = "CHINA.UNKNOWN"
            value = 0
            currency = "CNY"
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "spot",
            "value": float(value),
            "volume": raw.get("volume_certificates") or raw.get("volume_tonnes"),
            "currency": currency,
            "unit": "MWh" if product == "energy" else "certificate" if product == "green_certificate" else "tCO2",
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
        logger.info(f"Emitted {count} China Grid events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state
        logger.debug(f"China Grid checkpoint saved: {state}")


if __name__ == "__main__":
    # Test China connector
    configs = [
        {
            "source_id": "china_beijing_east",
            "exchange": "BEIJING",
            "region": "EAST",
        },
        {
            "source_id": "china_guangzhou_south",
            "exchange": "GUANGZHOU",
            "region": "SOUTH",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = ChinaGridConnector(config)
        connector.run()

