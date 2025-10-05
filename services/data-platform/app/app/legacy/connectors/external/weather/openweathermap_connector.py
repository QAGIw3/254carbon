"""
OpenWeatherMap Connector

Coverage: Global current weather and forecasts.
Portal: https://openweathermap.org/api

This scaffold emits sample current conditions; production integrates with the
OWM API using an API key (free tier available).
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class OpenWeatherMapConnector(Ingestor):
    """OpenWeatherMap connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # OWM REST base: https://api.openweathermap.org/data/2.5
        # Examples: /weather?lat=..&lon=..&appid=KEY&units=metric, /onecall?lat=..&lon=..&exclude=minutely&appid=KEY
        self.api_base = config.get("api_base", "https://api.openweathermap.org/data/2.5")
        self.api_key = config.get("api_key")
        self.location = config.get("location", "GLOB")  # city or coordinates in prod
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "current_weather", "market": "weather", "product": "temp_c", "unit": "C", "update_freq": "hourly"},
                {"name": "wind_speed", "market": "weather", "product": "wind_ms", "unit": "m/s", "update_freq": "hourly"},
                {"name": "precipitation", "market": "weather", "product": "precip_mm", "unit": "mm", "update_freq": "hourly"},
            ],
            "endpoint_examples": {
                "current": "GET {base}/weather?lat=34.05&lon=-118.24&appid=...&units=metric",
                "onecall": "GET {base}/onecall?lat=34.05&lon=-118.24&exclude=minutely&appid=...&units=metric",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        yield {"timestamp": now, "entity": self.location, "variable": "temp_c", "value": 19.2, "unit": "C"}
        yield {"timestamp": now, "entity": self.location, "variable": "wind_ms", "value": 3.6, "unit": "m/s"}
        yield {"timestamp": now, "entity": self.location, "variable": "precip_mm", "value": 0.0, "unit": "mm"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "GLOB")
        variable = raw.get("variable", "temp_c")
        instrument = f"OWM.{entity}.{variable.upper()}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "weather",
            "product": variable,
            "instrument_id": instrument,
            "location_code": instrument,
            "price_type": "observation",
            "value": float(raw.get("value", 0.0)),
            "volume": None,
            "currency": "USD",
            "unit": raw.get("unit", "unit"),
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }

    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
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
        if self.producer is not None:
            self.producer.flush()
        logger.info(f"Emitted {count} OpenWeatherMap events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"OpenWeatherMap checkpoint saved: {state}")


if __name__ == "__main__":
    connector = OpenWeatherMapConnector({"source_id": "openweathermap"})
    connector.run()
