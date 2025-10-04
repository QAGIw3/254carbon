"""Helpers for computing HDD/CDD metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict


@dataclass
class RegionTemperatureProfile:
    region: str
    base_temperature_f: float
    seasonal_amplitude: float
    phase_shift: float


REGION_TEMPERATURE_PROFILES: Dict[str, RegionTemperatureProfile] = {
    "PJM": RegionTemperatureProfile("PJM", base_temperature_f=52.0, seasonal_amplitude=22.0, phase_shift=0.0),
    "ERCOT": RegionTemperatureProfile("ERCOT", base_temperature_f=68.0, seasonal_amplitude=18.0, phase_shift=0.5),
    "NYISO": RegionTemperatureProfile("NYISO", base_temperature_f=50.0, seasonal_amplitude=24.0, phase_shift=0.1),
    "MIDWEST": RegionTemperatureProfile("MIDWEST", base_temperature_f=48.0, seasonal_amplitude=26.0, phase_shift=0.2),
}


def estimate_temperature(region: str, as_of: date) -> float:
    profile = REGION_TEMPERATURE_PROFILES.get(region.upper())
    if not profile:
        profile = RegionTemperatureProfile(region.upper(), 60.0, 20.0, 0.0)
    day_of_year = as_of.timetuple().tm_yday
    seasonal = profile.seasonal_amplitude * math.sin(2 * math.pi * (day_of_year / 365.0 + profile.phase_shift))
    return profile.base_temperature_f + seasonal


def compute_degree_days(region: str, as_of: date, base_temperature: float = 65.0) -> Dict[str, float]:
    temp = estimate_temperature(region, as_of)
    hdd = max(base_temperature - temp, 0.0)
    cdd = max(temp - base_temperature, 0.0)
    return {"hdd": hdd, "cdd": cdd, "temperature": temp}
