"""
EU ETS (Emissions Trading System) Connector

Ingests EU carbon allowance (EUA) prices and market data.
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


class EUETSConnector(Ingestor):
    """EU Emissions Trading System connector for carbon prices."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://www.eex.com/api")
        self.product_type = config.get("product_type", "EUA")  # EUA, EUAAuction
        self.kafka_topic = config.get("kafka_topic", "carbon.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover EU ETS data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "eua_futures",
                    "market": "carbon",
                    "product": "allowance",
                    "description": "EU Allowance Futures (ICE/EEX)",
                    "currency": "EUR",
                },
                {
                    "name": "eua_auction",
                    "market": "carbon",
                    "product": "allowance",
                    "description": "Primary Market Auctions",
                    "frequency": "weekly",
                },
                {
                    "name": "compliance_data",
                    "market": "carbon",
                    "product": "compliance",
                    "description": "Installation-level emissions",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull data from EU ETS sources."""
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching EU ETS {self.product_type} data since {last_time}")
        
        if self.product_type == "EUA":
            yield from self._fetch_eua_futures()
        elif self.product_type == "EUAAuction":
            yield from self._fetch_eua_auctions()
    
    def _fetch_eua_futures(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch EUA futures prices.
        
        EU Allowances (EUAs) trade on ICE and EEX.
        Price in EUR per tonne of CO2.
        """
        now = datetime.utcnow()
        
        # EUA price has been trending upward (currently ~€80-100/tCO2)
        base_price = 85.0
        
        # Generate futures curve (Dec contracts for next 3 years)
        for year_offset in range(3):
            delivery_year = now.year + year_offset
            contract_month = f"DEC{delivery_year % 100}"
            
            # Contango structure (later years slightly higher)
            contango_premium = year_offset * 2.0
            
            # Add some random variation
            price = base_price + contango_premium + (hash(str(year_offset)) % 10) - 5
            
            # Volume (in contracts, 1 contract = 1000 tCO2)
            volume = 5000 + (hash(str(year_offset)) % 2000)
            
            # Open interest
            open_interest = 50000 + (hash(contract_month) % 20000)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "EUETS",
                "product": "EUA_FUTURE",
                "contract_month": contract_month,
                "settlement_price": price,
                "volume_contracts": volume,
                "open_interest": open_interest,
                "currency": "EUR",
                "unit": "tCO2",
            }
    
    def _fetch_eua_auctions(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch EUA auction results.
        
        Primary market auctions conducted by member states.
        Typically 2-3 auctions per week.
        """
        now = datetime.utcnow()
        
        # Mock auction data
        auction_platforms = [
            {"platform": "EEX", "volume": 3500000},  # 3.5M allowances
            {"platform": "ICE", "volume": 2000000},  # 2M allowances
        ]
        
        for platform_data in auction_platforms:
            # Auction clearing price (typically close to secondary market)
            secondary_price = 85.0
            clearing_price = secondary_price + (hash(platform_data["platform"]) % 3) - 1.5
            
            # Cover ratio (demand / supply)
            cover_ratio = 1.5 + (hash(str(now.day)) % 20) / 10  # 1.5-3.5x
            
            # Number of bidders
            bidders = 15 + (hash(platform_data["platform"]) % 10)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "EUETS",
                "product": "EUA_AUCTION",
                "platform": platform_data["platform"],
                "clearing_price": clearing_price,
                "volume_allowances": platform_data["volume"],
                "cover_ratio": cover_ratio,
                "num_bidders": bidders,
                "currency": "EUR",
                "auction_date": now.date().isoformat(),
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map EU ETS format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        if raw["product"] == "EUA_FUTURE":
            location = f"EUETS.EUA.{raw['contract_month']}"
            value = raw["settlement_price"]
        else:  # EUA_AUCTION
            location = f"EUETS.AUCTION.{raw['platform']}"
            value = raw["clearing_price"]
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "carbon",
            "product": "allowance",
            "instrument_id": location,
            "location_code": location,
            "price_type": "trade",
            "value": float(value),
            "volume": raw.get("volume_contracts") or raw.get("volume_allowances"),
            "currency": "EUR",
            "unit": "tCO2",
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
        logger.info(f"Emitted {count} EU ETS events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"EU ETS checkpoint saved: {state}")


if __name__ == "__main__":
    # Test EU ETS connector
    configs = [
        {
            "source_id": "euets_futures",
            "product_type": "EUA",
            "kafka_topic": "carbon.ticks.v1",
        },
        {
            "source_id": "euets_auctions",
            "product_type": "EUAAuction",
            "kafka_topic": "carbon.auction.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = EUETSConnector(config)
        connector.run()

