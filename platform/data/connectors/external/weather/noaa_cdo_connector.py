"""
NOAA Climate Data Online (CDO) Connector

Coverage: US/global weather, climate, severe events.
Portal: https://www.ncdc.noaa.gov/cdo-web/

Production notes: Use token-based API (https://www.ncdc.noaa.gov/cdo-web/webservices/v2)
with dataset IDs like GHCND, GSOM, GSOY etc. This scaffold emits mocked
observations for temperature, wind, and precipitation.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, Optional, List

from kafka import KafkaProducer
import json
import requests

from ...base import Ingestor

logger = logging.getLogger(__name__)


class NOAACDOConnector(Ingestor):
    """NOAA CDO connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # NOAA CDO REST base: https://www.ncdc.noaa.gov/cdo-web/api/v2
        # Endpoints: /datasets, /datatypes, /stations, /data
        # Example: GET /data?datasetid=GHCND&stationid=GHCND:USW00023174&startdate=2024-01-01&enddate=2024-01-31
        # Auth: header 'token: <YOUR_TOKEN>'
        self.api_base = config.get("api_base", "https://www.ncdc.noaa.gov/cdo-web/api/v2")
        self.token = config.get("token")
        self.location = config.get("location", "GLOB")
        # Live fetch options
        self.live: bool = config.get("live", False)
        self.datasetid: str = config.get("datasetid", "GHCND")
        self.stationid: Optional[str] = config.get("stationid")  # e.g., GHCND:USW00023174
        self.locationid: Optional[str] = config.get("locationid")  # e.g., FIPS:06
        dtype = config.get("datatypeid", ["TAVG", "PRCP", "AWND"])  # list or str
        self.datatypeid: List[str] = dtype if isinstance(dtype, list) else [dtype]
        self.startdate: Optional[str] = config.get("startdate")  # YYYY-MM-DD
        self.enddate: Optional[str] = config.get("enddate")
        self.limit: int = int(config.get("limit", 1000))
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "temperature_daily", "market": "weather", "product": "temp_mean_c", "unit": "C", "update_freq": "daily"},
                {"name": "wind_speed", "market": "weather", "product": "wind_ms", "unit": "m/s", "update_freq": "hourly"},
                {"name": "precipitation", "market": "weather", "product": "precip_mm", "unit": "mm", "update_freq": "daily"},
            ],
            "endpoint_examples": {
                "datasets": "GET {base}/datasets (header token)",
                "stations": "GET {base}/stations?datasetid=GHCND&locationid=FIPS:06 (header token)",
                "daily_data": (
                    "GET {base}/data?datasetid=GHCND&datatypeid=TMIN&stationid=GHCND:USW00023174"
                    "&startdate=2024-01-01&enddate=2024-01-31&limit=1000"
                ),
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self.location, "variable": "temp_mean_c", "value": 16.7, "unit": "C"}
            yield {"timestamp": now, "entity": self.location, "variable": "wind_ms", "value": 4.2, "unit": "m/s"}
            yield {"timestamp": now, "entity": self.location, "variable": "precip_mm", "value": 2.4, "unit": "mm"}
            return

        if not self.token:
            raise ValueError("NOAA CDO live mode requires 'token' in config")

        if not (self.stationid or self.locationid):
            raise ValueError("Provide 'stationid' or 'locationid' for NOAA CDO query")

        startdate = self.startdate or f"{datetime.utcnow().year}-01-01"
        enddate = self.enddate or datetime.utcnow().strftime("%Y-%m-%d")

        url = f"{self.api_base}/data"
        params = {
            "datasetid": self.datasetid,
            "startdate": startdate,
            "enddate": enddate,
            "limit": self.limit,
        }
        for dt in self.datatypeid:
            params.setdefault("datatypeid", []).append(dt)
        if self.stationid:
            params["stationid"] = self.stationid
        if self.locationid:
            params["locationid"] = self.locationid

        headers = {"token": self.token}

        # Pagination loop
        offset = 1
        max_pages = 50  # safety guard
        page_count = 0
        while True:
            page_count += 1
            if page_count > max_pages:
                logging.warning("NOAA CDO pagination limit reached; stopping early")
                break

            params["offset"] = offset
            try:
                r = requests.get(url, headers=headers, params=params, timeout=30)
                r.raise_for_status()
                payload = r.json()
            except Exception as e:
                logging.error(f"NOAA CDO request failed (offset {offset}): {e}")
                break

            results = (payload or {}).get("results", [])
            if not results:
                break

            # Results are list of observations with fields: date, value, datatype
            for row in results:
                date_str = row.get("date")  # e.g., 2024-01-01T00:00:00
                datatype = row.get("datatype")
                value = row.get("value")
                if date_str is None or value is None:
                    continue
            # Map datatypes to canonical variables
            if datatype == "PRCP":
                var, unit = "precip_mm", "mm"
            elif datatype in ("TAVG", "TMAX", "TMIN"):
                var, unit = "temp_c", "C"
            elif datatype in ("AWND", "WSF2", "WSF5"):
                var, unit = "wind_ms", "m/s"
            else:
                var, unit = datatype.lower(), "unit"
            yield {"timestamp": date_str.replace("Z", ""), "entity": self.location, "variable": var, "value": float(value), "unit": unit}

            # Prepare next page
            if len(results) < self.limit:
                break
            offset += self.limit

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "GLOB")
        variable = raw.get("variable", "temp_mean_c")
        instrument = f"NOAA.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} NOAA CDO events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"NOAA CDO checkpoint saved: {state}")


if __name__ == "__main__":
    connector = NOAACDOConnector({"source_id": "noaa_cdo"})
    connector.run()
