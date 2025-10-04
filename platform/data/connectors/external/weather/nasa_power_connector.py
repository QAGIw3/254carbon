"""
NASA POWER Connector

Coverage: Solar radiation, wind, temperature, climate variables.
Portal: https://power.larc.nasa.gov/

Production: Use the REST API with community parameters; this scaffold emits
representative solar GHI, wind speed, and ambient temperature observations.

Data Flow
---------
NASA POWER API (or mocks) → parse hourly/daily point data → canonical series → Kafka
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

from kafka import KafkaProducer
import json
import requests

from ...base import Ingestor

logger = logging.getLogger(__name__)


class NASAPowerConnector(Ingestor):
    """NASA POWER connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # NASA POWER REST examples:
        # Hourly point: /api/temporal/hourly/point?parameters=T2M,WS10M,ALLSKY_SFC_SW_DWN&latitude=..&longitude=..&start=YYYYMMDD&end=YYYYMMDD&community=AG&format=JSON
        # Daily point:  /api/temporal/daily/point?parameters=T2M_MAX,T2M_MIN&latitude=..&longitude=..&start=YYYYMMDD&end=YYYYMMDD&community=AG&format=JSON
        self.api_base = config.get("api_base", "https://power.larc.nasa.gov/api")
        self.site = config.get("site", "GLOB")
        self.latitude: Optional[float] = config.get("latitude")
        self.longitude: Optional[float] = config.get("longitude")
        self.temporal: str = config.get("temporal", "hourly")  # hourly or daily
        self.parameters: str = config.get("parameters", "T2M,WS10M,ALLSKY_SFC_SW_DWN")
        self.start: Optional[str] = config.get("start")  # YYYYMMDD
        self.end: Optional[str] = config.get("end")      # YYYYMMDD
        self.community: str = config.get("community", "AG")
        self.live: bool = config.get("live", False)
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "solar_ghi", "market": "weather", "product": "ghi_wm2", "unit": "W/m2", "update_freq": "hourly"},
                {"name": "wind_speed", "market": "weather", "product": "wind_ms", "unit": "m/s", "update_freq": "hourly"},
                {"name": "temperature", "market": "weather", "product": "temp_c", "unit": "C", "update_freq": "hourly"},
            ],
            "endpoint_examples": {
                "hourly_point": (
                    "GET {base}/temporal/hourly/point?parameters=T2M,WS10M,ALLSKY_SFC_SW_DWN"
                    "&latitude=34.05&longitude=-118.24&start=20240101&end=20240102&community=AG&format=JSON"
                ),
                "daily_point": (
                    "GET {base}/temporal/daily/point?parameters=T2M_MAX,T2M_MIN"
                    "&latitude=34.05&longitude=-118.24&start=20240101&end=20240131&community=AG&format=JSON"
                ),
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self.site, "variable": "ghi_mj_m2_hr", "value": 1.6, "unit": "MJ/m2/hr"}
            yield {"timestamp": now, "entity": self.site, "variable": "wind_ms", "value": 4.8, "unit": "m/s"}
            yield {"timestamp": now, "entity": self.site, "variable": "temp_c", "value": 21.3, "unit": "C"}
            return

        # Live mode: build request and fetch JSON
        if self.latitude is None or self.longitude is None:
            raise ValueError("latitude and longitude are required in live mode")

        temporal = self.temporal.lower()
        if temporal not in ("hourly", "daily"):
            temporal = "hourly"

        # Default to one-day window (UTC today)
        today = datetime.utcnow().strftime("%Y%m%d")
        start = self.start or today
        end = self.end or today

        url = (
            f"{self.api_base}/temporal/{temporal}/point?parameters={self.parameters}"
            f"&latitude={self.latitude}&longitude={self.longitude}"
            f"&start={start}&end={end}&community={self.community}&format=JSON"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"NASA POWER request failed: {e}")
            # Fallback to single mock sample
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self.site, "variable": "ghi_mj_m2_hr", "value": 1.2, "unit": "MJ/m2/hr"}
            return

        # Parse parameters dict
        params = (data or {}).get("properties", {}).get("parameter", {})
        def iter_series(var_key: str, variable_name: str, unit: str):
            series = params.get(var_key, {})
            for ts_str, value in series.items():
                # ts_str like YYYYMMDDHH for hourly or YYYYMMDD for daily (LST)
                try:
                    if len(ts_str) == 10:  # hourly
                        dt = datetime.strptime(ts_str, "%Y%m%d%H")
                    else:
                        dt = datetime.strptime(ts_str, "%Y%m%d")
                    ts_iso = dt.isoformat()
                except Exception:
                    ts_iso = datetime.utcnow().isoformat()
                yield {"timestamp": ts_iso, "entity": self.site, "variable": variable_name, "value": float(value), "unit": unit}

        # T2M (C)
        yield from iter_series("T2M", "temp_c", "C")
        # WS10M (m/s)
        yield from iter_series("WS10M", "wind_ms", "m/s")
        # ALLSKY_SFC_SW_DWN (MJ/m2/hr)
        yield from iter_series("ALLSKY_SFC_SW_DWN", "ghi_mj_m2_hr", "MJ/m2/hr")

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "GLOB")
        variable = raw.get("variable", "ghi_mj_m2_hr")
        instrument = f"NASA_POWER.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} NASA POWER events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"NASA POWER checkpoint saved: {state}")


if __name__ == "__main__":
    connector = NASAPowerConnector({"source_id": "nasa_power"})
    connector.run()
