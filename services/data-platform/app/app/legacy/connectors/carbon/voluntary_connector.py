"""
Voluntary Carbon Markets Connector

Ingests voluntary carbon credit prices including:
- Nature-based solutions (forestry, wetlands)
- Technology-based (DAC, biochar)
- Project quality ratings
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

from kafka import KafkaProducer
import json

from ..base import Ingestor

logger = logging.getLogger(__name__)


class VoluntaryCarbonConnector(Ingestor):
    """Voluntary carbon markets connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.project_type = config.get("project_type", "ALL")
        self.kafka_topic = config.get("kafka_topic", "carbon.voluntary.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover voluntary carbon market streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "nbs_credits",
                    "description": "Nature-based solution credits",
                    "project_types": ["forestry", "wetlands", "soil_carbon", "blue_carbon"],
                },
                {
                    "name": "tech_credits",
                    "description": "Technology-based credits",
                    "project_types": ["dac", "biochar", "enhanced_weathering", "ccus"],
                },
                {
                    "name": "project_ratings",
                    "description": "Independent quality assessments",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull voluntary market data."""
        logger.info(f"Fetching voluntary carbon credits - {self.project_type}")
        yield from self._fetch_vcm_prices()
    
    def _fetch_vcm_prices(self) -> Iterator[Dict[str, Any]]:
        """Fetch VCM prices by project type."""
        # Project types with typical pricing
        projects = [
            {"type": "forestry_redd", "price": 12.50, "quality": "high", "additionality": 0.85},
            {"type": "forestry_afforestation", "price": 15.00, "quality": "high", "additionality": 0.90},
            {"type": "wetlands_restoration", "price": 18.00, "quality": "high", "additionality": 0.88},
            {"type": "soil_carbon", "price": 10.00, "quality": "medium", "additionality": 0.70},
            {"type": "blue_carbon_mangrove", "price": 22.00, "quality": "high", "additionality": 0.92},
            {"type": "dac", "price": 250.00, "quality": "high", "additionality": 1.00},
            {"type": "biochar", "price": 120.00, "quality": "high", "additionality": 0.95},
            {"type": "enhanced_weathering", "price": 80.00, "quality": "medium", "additionality": 0.75},
            {"type": "ccus_industrial", "price": 60.00, "quality": "high", "additionality": 0.90},
        ]
        
        for project in projects:
            if self.project_type != "ALL" and self.project_type not in project["type"]:
                continue
            
            # Add price variation
            price = project["price"] * (1 + np.random.normal(0, 0.10))
            price = max(5.0, price)
            
            # Volume varies
            volume = 10000 + int(np.random.normal(0, 3000))
            volume = max(1000, volume)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "VCM",
                "project_type": project["type"],
                "price_usd_tco2": price,
                "volume_credits": volume,
                "quality_rating": project["quality"],
                "additionality_score": project["additionality"],
                "co_benefits": ["biodiversity", "community"] if "forestry" in project["type"] or "wetlands" in project["type"] else [],
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map VCM format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        location = f"VCM.{raw['project_type'].upper()}"
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "carbon",
            "product": "voluntary_credit",
            "instrument_id": location,
            "location_code": location,
            "price_type": "trade",
            "value": float(raw["price_usd_tco2"]),
            "volume": raw.get("volume_credits"),
            "currency": "USD",
            "unit": "tCO2e",
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
        logger.info(f"Emitted {count} VCM events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint."""
        self.checkpoint_state = state


if __name__ == "__main__":
    connector = VoluntaryCarbonConnector({"source_id": "vcm_global"})
    connector.run()

