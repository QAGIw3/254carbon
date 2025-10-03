"""
MISO Nodal LMP Connector

Pulls real-time and day-ahead LMP data from MISO OASIS API.
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


class MISOConnector(Ingestor):
    """MISO nodal DA/RT LMP connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://api.misoenergy.org/MISORTWDDataBroker")
        self.market_type = config.get("market_type", "RT")  # RT or DA
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover MISO nodes and pricing points."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "nodal_lmp",
                    "market": "power",
                    "product": "lmp",
                    "update_freq": "5min" if self.market_type == "RT" else "1hour",
                    "nodes": "~3000",  # MISO has ~3000 pricing nodes
                }
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull LMP data from MISO OASIS.
        
        For RT: polls every 5 minutes
        For DA: polls hourly for next day
        """
        last_checkpoint = self.load_checkpoint()
        last_time = self._resolve_last_time(last_checkpoint)
        
        # In production, this would use actual MISO OASIS API
        # For now, simulate with mock data
        logger.info(f"Fetching MISO {self.market_type} LMP since {last_time}")
        
        # Mock: generate sample data
        # Real implementation would call:
        # response = requests.get(f"{self.api_base}/lmp", params={...})
        
        # Simulate nodal data
        nodes = [f"MISO.NODE.{i:04d}" for i in range(1, 51)]  # Sample 50 nodes
        
        for node in nodes:
            event_time = datetime.now(timezone.utc)
            yield {
                "timestamp": event_time.isoformat(),
                "node_id": node,
                "lmp": 35.50 + (hash(node) % 100) / 10,  # Mock price
                "mcc": 2.30,  # Congestion
                "mlc": 1.20,  # Loss
                "market": self.market_type,
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map MISO format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),  # milliseconds
            "market": "power",
            "product": "lmp",
            "instrument_id": raw["node_id"],
            "location_code": raw["node_id"],
            "price_type": "settle" if raw["market"] == "DA" else "trade",
            "value": float(raw["lmp"]),
            "volume": None,
            "currency": "USD",
            "unit": "MWh",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }

    def _resolve_last_time(self, checkpoint: Optional[Dict[str, Any]]) -> datetime:
        """Resolve the reference timestamp for incremental pulling."""
        if not checkpoint:
            return datetime.now(timezone.utc) - timedelta(hours=1)

        last_event_time = checkpoint.get("last_event_time")

        if last_event_time is None:
            return datetime.now(timezone.utc) - timedelta(hours=1)

        if isinstance(last_event_time, (int, float)):
            return datetime.fromtimestamp(last_event_time / 1000, tz=timezone.utc)

        if isinstance(last_event_time, str):
            try:
                return datetime.fromisoformat(last_event_time)
            except ValueError:
                logger.warning("Invalid last_event_time in checkpoint, defaulting to 1 hour lookback")
                return datetime.now(timezone.utc) - timedelta(hours=1)

        if isinstance(last_event_time, datetime):
            return last_event_time.astimezone(timezone.utc)

        logger.warning("Unsupported last_event_time type in checkpoint, defaulting to 1 hour lookback")
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
        logger.info(f"Emitted {count} events to {self.kafka_topic}")
        return count
    


if __name__ == "__main__":
    # Test connector
    config = {
        "source_id": "miso_rt_lmp",
        "market_type": "RT",
        "kafka_topic": "power.ticks.v1",
    }
    
    connector = MISOConnector(config)
    connector.run()

