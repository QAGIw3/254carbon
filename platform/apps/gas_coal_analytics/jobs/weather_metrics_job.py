"""HDD/CDD metric generation job backed by live weather feeds."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd

from platform.data.connectors.external.weather.noaa_cdo_connector import (
    NOAACDOConnector,
)
from platform.data.connectors.external.weather.nasa_power_connector import (
    NASAPowerConnector,
)

from ..clients.clickhouse import insert_rows

logger = logging.getLogger(__name__)

BASE_TEMPERATURE_F = 65.0
LOOKBACK_DAYS = 3


@dataclass
class RegionWeatherConfig:
    """Connector configuration for a region."""

    noaa_station: Optional[str] = None
    noaa_location: Optional[str] = None
    nasa_latitude: Optional[float] = None
    nasa_longitude: Optional[float] = None
    base_temperature_f: float = BASE_TEMPERATURE_F


REGION_WEATHER_CONFIG: Dict[str, RegionWeatherConfig] = {
    "PJM": RegionWeatherConfig(
        noaa_station="GHCND:USW00013739",  # Philadelphia International Airport
        nasa_latitude=39.8719,
        nasa_longitude=-75.2411,
    ),
    "ERCOT": RegionWeatherConfig(
        noaa_station="GHCND:USW00013958",  # Houston Intercontinental Airport
        nasa_latitude=29.9902,
        nasa_longitude=-95.3368,
    ),
    "NYISO": RegionWeatherConfig(
        noaa_station="GHCND:USW00014732",  # New York Central Park
        nasa_latitude=40.7794,
        nasa_longitude=-73.9692,
    ),
    "MIDWEST": RegionWeatherConfig(
        noaa_station="GHCND:USW00094846",  # Chicago O'Hare
        nasa_latitude=41.9742,
        nasa_longitude=-87.9073,
    ),
    "HENRY": RegionWeatherConfig(
        noaa_location="FIPS:22",  # Louisiana
        nasa_latitude=29.9997,
        nasa_longitude=-92.1003,
    ),
    "DAWN": RegionWeatherConfig(
        noaa_location="FIPS:36",  # New York state proxy
        nasa_latitude=42.9886,
        nasa_longitude=-78.9773,
    ),
}


def _parse_temperature_events(events: Iterable[Dict[str, object]]) -> pd.Series:
    """Convert connector events to a daily Fahrenheit temperature series."""

    records: List[Dict[str, object]] = []
    for event in events:
        variable = str(event.get("variable", "")).lower()
        if variable not in {"temp_c", "temp_mean_c", "temperature"}:
            continue
        timestamp_raw = event.get("timestamp")
        value = event.get("value")
        if timestamp_raw is None or value is None:
            continue
        try:
            ts = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00"))
        except ValueError:
            logger.debug("Skipping weather record with invalid timestamp: %s", timestamp_raw)
            continue
        try:
            temperature_c = float(value)
        except (TypeError, ValueError):
            logger.debug("Skipping weather record with non-numeric temperature: %s", value)
            continue
        if abs(temperature_c) > 120:
            # NOAA CDO returns tenths of °C for GHCND datasets.
            temperature_c = temperature_c / 10.0
        unit = str(event.get("unit", "")).lower()
        if "f" in unit and "c" not in unit:
            temperature_f = temperature_c
        else:
            temperature_f = temperature_c * 9.0 / 5.0 + 32.0
        records.append({"date": ts.date(), "temp_f": temperature_f})
    if not records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(records)
    grouped = df.groupby("date")["temp_f"].mean()
    grouped.index = pd.to_datetime(grouped.index)
    return grouped.sort_index()


def _fetch_noaa_series(region: str, cfg: RegionWeatherConfig, as_of: date) -> pd.Series:
    if not (cfg.noaa_station or cfg.noaa_location):
        return pd.Series(dtype=float)
    token = os.getenv("NOAA_CDO_TOKEN")
    live_default = os.getenv("NOAA_CDO_LIVE", "true").lower() == "true"
    live = bool(token) and live_default
    start = (as_of - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")
    config = {
        "source_id": f"noaa_{region.lower()}",
        "datasetid": "GHCND",
        "datatypeid": ["TAVG", "TMAX", "TMIN"],
        "startdate": start,
        "enddate": end,
        "limit": 1000,
        "live": live,
    }
    if live:
        config["token"] = token
    if cfg.noaa_station:
        config["stationid"] = cfg.noaa_station
    if cfg.noaa_location:
        config["locationid"] = cfg.noaa_location
    connector = NOAACDOConnector(config)
    events = list(connector.pull_or_subscribe())
    return _parse_temperature_events(events)


def _fetch_nasa_series(region: str, cfg: RegionWeatherConfig, as_of: date) -> pd.Series:
    if cfg.nasa_latitude is None or cfg.nasa_longitude is None:
        return pd.Series(dtype=float)
    live = os.getenv("NASA_POWER_LIVE", "false").lower() == "true"
    start = (as_of - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    end = as_of.strftime("%Y%m%d")
    config = {
        "source_id": f"nasa_power_{region.lower()}",
        "latitude": cfg.nasa_latitude,
        "longitude": cfg.nasa_longitude,
        "temporal": "hourly",
        "parameters": "T2M",
        "start": start,
        "end": end,
        "live": live,
        "site": region.upper(),
    }
    connector = NASAPowerConnector(config)
    events = list(connector.pull_or_subscribe())
    return _parse_temperature_events(events)


def _resolve_temperature(region: str, cfg: RegionWeatherConfig, as_of: date) -> Optional[float]:
    series_noaa = _fetch_noaa_series(region, cfg, as_of)
    series_nasa = _fetch_nasa_series(region, cfg, as_of)
    target = pd.Timestamp(as_of)

    temps: List[float] = []
    if target in series_noaa.index:
        temps.append(float(series_noaa.loc[target]))
    if target in series_nasa.index:
        temps.append(float(series_nasa.loc[target]))
    if not temps:
        combined = pd.concat([series_noaa, series_nasa]).sort_index()
        if not combined.empty:
            temps.append(float(combined.iloc[-1]))
    if not temps:
        logger.warning("No temperature observations available for %s on %s", region, as_of)
        return None
    return sum(temps) / len(temps)


def run_hdd_cdd_metrics_job(as_of: date, regions: Iterable[str]) -> List[dict]:
    rows: List[dict] = []
    for region in regions:
        region_key = region.upper()
        cfg = REGION_WEATHER_CONFIG.get(region_key, RegionWeatherConfig())
        temp_f = _resolve_temperature(region_key, cfg, as_of)
        if temp_f is None:
            continue
        base = cfg.base_temperature_f
        hdd = max(base - temp_f, 0.0)
        cdd = max(temp_f - base, 0.0)
        region_rows = [
            {
                "date": as_of,
                "entity_id": region_key,
                "instrument_id": None,
                "metric_name": "hdd",
                "metric_value": hdd,
                "unit": "degF_day",
                "version": "v1",
            },
            {
                "date": as_of,
                "entity_id": region_key,
                "instrument_id": None,
                "metric_name": "cdd",
                "metric_value": cdd,
                "unit": "degF_day",
                "version": "v1",
            },
        ]
        rows.extend(region_rows)
        logger.info(
            "Computed HDD/CDD for %s on %s using %.2f°F (HDD %.2f, CDD %.2f)",
            region_key,
            as_of,
            temp_f,
            hdd,
            cdd,
        )
    if rows:
        insert_rows("market_intelligence.supply_demand_metrics", rows)
        logger.info("Inserted %d HDD/CDD rows for %s", len(rows), as_of)
    return rows
