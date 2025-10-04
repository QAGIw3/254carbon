"""
Copernicus Climate Data Store (CDS) Connector

Coverage: Europe/global satellite, reanalysis, forecasts, climate projections.
Portal: https://cds.climate.copernicus.eu/

Live mode uses cdsapi to download NetCDF (e.g., ERA5 single levels) and parse
hourly series for canonical variables such as t2m (C), 10m wind (m/s), and
precipitation (mm). Falls back to safe mocks when cdsapi/netCDF4 are missing.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

from kafka import KafkaProducer
import json

from ...base import Ingestor

logger = logging.getLogger(__name__)


class CopernicusCDSConnector(Ingestor):
    """Copernicus CDS connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # cdsapi client config (can also use ~/.cdsapirc)
        self.cds_url: Optional[str] = config.get("cds_url")
        self.cds_key: Optional[str] = config.get("cds_key")

        # Dataset and request parameters
        self.dataset: str = config.get("dataset", "reanalysis-era5-single-levels")
        self.variables: List[str] = config.get(
            "variables",
            [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "total_precipitation",
            ],
        )
        self.area: Optional[List[float]] = config.get("area")
        self.point: Optional[Tuple[float, float]] = None
        if config.get("latitude") is not None and config.get("longitude") is not None:
            self.point = (float(config["latitude"]), float(config["longitude"]))
        self.start_date: str = config.get("start_date", datetime.utcnow().strftime("%Y-%m-%d"))
        self.end_date: str = config.get("end_date", self.start_date)
        self.hours: List[str] = config.get("hours", [f"{h:02d}:00" for h in range(24)])
        self.file_format: str = config.get("format", "netcdf")
        self.live: bool = bool(config.get("live", False))

        # Emission
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "t2m", "market": "weather", "product": "t2m_c", "unit": "C", "update_freq": "hourly"},
                {"name": "winds10m", "market": "weather", "product": "wind_ms", "unit": "m/s", "update_freq": "hourly"},
                {"name": "tp", "market": "weather", "product": "precip_mm", "unit": "mm", "update_freq": "hourly"},
            ],
            "endpoint_examples": {
                "cdsapi_python": (
                    "cdsapi.Client().retrieve('reanalysis-era5-single-levels', {variable:['2m_temperature'], ...}, 'out.nc')"
                ),
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self._entity_label(), "variable": "t2m_c", "value": 15.8, "unit": "C"}
            yield {"timestamp": now, "entity": self._entity_label(), "variable": "precip_mm", "value": 1.1, "unit": "mm"}
            yield {"timestamp": now, "entity": self._entity_label(), "variable": "wind_ms", "value": 5.1, "unit": "m/s"}
            return

        try:
            import cdsapi  # type: ignore
            import tempfile
            import os
            from netCDF4 import Dataset, num2date  # type: ignore
        except Exception as e:
            logger.error(f"CDS live mode requires cdsapi and netCDF4 installed: {e}")
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self._entity_label(), "variable": "t2m_c", "value": 15.0, "unit": "C"}
            return

        # Build client (optional explicit creds)
        client_kwargs: Dict[str, Any] = {}
        if self.cds_url and self.cds_key:
            client_kwargs = {"url": self.cds_url, "key": self.cds_key}
        client = cdsapi.Client(**client_kwargs)

        payload: Dict[str, Any] = {
            "product_type": "reanalysis",
            "variable": self.variables,
            "year": list({self.start_date[:4], self.end_date[:4]}),
            "month": list({self.start_date[5:7], self.end_date[5:7]}),
            "day": list({self.start_date[8:10], self.end_date[8:10]}),
            "time": self.hours,
            "format": self.file_format,
        }
        if self.area:
            payload["area"] = self.area

        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "cds.nc")
            client.retrieve(self.dataset, payload, out_path)

            ds = Dataset(out_path)
            try:
                times = ds.variables["time"]
                time_vals = num2date(times[:], units=times.units, only_use_cftime_datetimes=False)

                def spatial_mean(varname: str, idx: int) -> float:
                    arr = ds.variables[varname][idx]
                    try:
                        return float(arr.mean())
                    except Exception:
                        return float(arr)

                for i, t in enumerate(time_vals):
                    ts_iso = t.isoformat()
                    if "t2m" in ds.variables:
                        k = spatial_mean("t2m", i)
                        yield {"timestamp": ts_iso, "entity": self._entity_label(), "variable": "t2m_c", "value": float(k - 273.15), "unit": "C"}
                    if "u10" in ds.variables and "v10" in ds.variables:
                        u = spatial_mean("u10", i)
                        v = spatial_mean("v10", i)
                        wind = float((u * u + v * v) ** 0.5)
                        yield {"timestamp": ts_iso, "entity": self._entity_label(), "variable": "wind_ms", "value": wind, "unit": "m/s"}
                    if "tp" in ds.variables:
                        m = spatial_mean("tp", i)
                        yield {"timestamp": ts_iso, "entity": self._entity_label(), "variable": "precip_mm", "value": float(m * 1000.0), "unit": "mm"}
            finally:
                ds.close()

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", self._entity_label())
        variable = raw.get("variable", "t2m_c")
        instrument = f"CDS.{entity}.{variable.upper()}"
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
        logger.info(f"Emitted {count} Copernicus CDS events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"Copernicus CDS checkpoint saved: {state}")

    def _entity_label(self) -> str:
        if self.point is not None:
            lat, lon = self.point
            return f"PT_{lat:.2f}_{lon:.2f}"
        if self.area is not None and isinstance(self.area, list) and len(self.area) == 4:
            n, w, s, e = self.area
            return f"AREA_{n:.2f}_{w:.2f}_{s:.2f}_{e:.2f}"
        return "GLOB"


if __name__ == "__main__":
    connector = CopernicusCDSConnector({
        "source_id": "copernicus_cds",
        "live": False,  # set True with cds credentials and dates to fetch live
    })
    connector.run()
