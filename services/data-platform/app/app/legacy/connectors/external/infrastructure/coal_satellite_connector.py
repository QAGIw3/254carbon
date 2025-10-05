"""
Satellite-derived coal stockpile connector.

Pulls site-level stockpile metrics from the `satellite-intel` service and
emits canonical fundamentals events. Designed to be orchestrated
via Airflow DAGs pulling configuration/environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

from ...base import CommodityType, Ingestor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CoalSite:
    site_id: str
    name: str
    country: str
    lat: float
    lon: float


@dataclass(slots=True)
class StockpileMeasurement:
    site: CoalSite
    measurement_date: datetime
    volume_tonnes: float
    change_pct_7d: Optional[float]
    area_hectares: Optional[float]
    average_height_m: Optional[float]
    confidence: Optional[float]


class CoalSatelliteConnector(Ingestor):
    """Connector to fetch coal stockpile analytics from satellite-intel service."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.COAL

        self.api_base = config.get("api_base", "http://satellite-intel:8025").rstrip("/")
        self.sites: List[CoalSite] = [
            CoalSite(**site) if isinstance(site, dict) else site
            for site in config.get("sites", [])
        ]
        self.timeout = float(config.get("timeout", 30.0))

        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "streams": [
                {
                    "name": "coal_stockpile",
                    "variables": ["coal_stockpile_tonnes", "coal_stockpile_change_7d_pct"],
                    "frequency": "daily",
                }
            ],
            "sites": [site.site_id for site in self.sites],
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        for site in self.sites:
            try:
                measurement = self._fetch_site_measurement(site)
            except Exception as exc:
                logger.error("Failed to fetch coal measurement for %s: %s", site.site_id, exc)
                continue
            yield from self._measurement_to_events(measurement)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts: datetime = raw["event_time"]
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "infra",
            "product": raw["product"],
            "instrument_id": raw["instrument_id"],
            "location_code": raw["location_code"],
            "price_type": "observation",
            "value": raw["value"],
            "unit": raw["unit"],
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "metadata": json.dumps(raw.get("metadata", {})),
        }

    def emit(self, events: Iterable[Dict[str, Any]]) -> int:
        return super().emit(events)

    def checkpoint(self, state: Dict[str, Any]) -> None:
        super().checkpoint(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_site_measurement(self, site: CoalSite) -> StockpileMeasurement:
        url = f"{self.api_base}/api/v1/satellite/coal-stockpile/{site.site_id}"
        params = {"lat": site.lat, "lon": site.lon}
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        measurement_date = datetime.fromisoformat(payload["measurement_date"]).replace(tzinfo=timezone.utc)
        return StockpileMeasurement(
            site=site,
            measurement_date=measurement_date,
            volume_tonnes=float(payload.get("volume_tonnes", 0.0)),
            change_pct_7d=self._safe_float(payload.get("change_7d_pct")),
            area_hectares=self._safe_float(payload.get("area_hectares")),
            average_height_m=self._safe_float(payload.get("average_height_meters")),
            confidence=self._safe_float(payload.get("confidence")),
        )

    def _measurement_to_events(self, measurement: StockpileMeasurement) -> Iterator[Dict[str, Any]]:
        site = measurement.site
        instrument_id = f"COAL.{site.site_id}.STOCKPILE"
        base_metadata = {
            "site_name": site.name,
            "country": site.country,
            "lat": site.lat,
            "lon": site.lon,
            "confidence": measurement.confidence,
        }

        yield {
            "event_time": measurement.measurement_date,
            "product": "coal_stockpile_tonnes",
            "value": measurement.volume_tonnes,
            "unit": "tonnes",
            "instrument_id": instrument_id,
            "location_code": site.site_id,
            "metadata": base_metadata,
        }

        if measurement.change_pct_7d is not None:
            yield {
                "event_time": measurement.measurement_date,
                "product": "coal_stockpile_change_7d_pct",
                "value": measurement.change_pct_7d,
                "unit": "pct",
                "instrument_id": instrument_id,
                "location_code": site.site_id,
                "metadata": base_metadata,
            }

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def load_sites_from_csv(csv_path: str) -> List[CoalSite]:
    sites: List[CoalSite] = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Coal sites CSV not found: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = __import__("csv").DictReader(fp)
        for row in reader:
            try:
                sites.append(
                    CoalSite(
                        site_id=row["site_id"],
                        name=row["name"],
                        country=row["country"],
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping invalid row %s: %s", row, exc)
    return sites
