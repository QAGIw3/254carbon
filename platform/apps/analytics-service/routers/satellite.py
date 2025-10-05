"""
Satellite Intelligence API Router

Purpose
-------
Serves satellite-derived intelligence (e.g., flaring, activity metrics) and
related analytics built by the `satellite` engines.
"""
import logging
from datetime import date
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from satellite.intelligence import SatelliteIntelligence, SatelliteProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/satellite", tags=["satellite"])

# Initialize intelligence engine
intelligence = SatelliteIntelligence()


class StorageTankMeasurement(BaseModel):
    """Oil storage tank floating roof measurement."""
    tank_id: str
    location: Dict[str, float]
    diameter_meters: float
    height_meters: float
    fill_level_pct: float
    volume_barrels: float
    change_since_last: float
    measurement_date: date
    confidence: float


class CoalStockpileMeasurement(BaseModel):
    """Coal stockpile volume estimate."""
    site_id: str
    location: Dict[str, float]
    volume_tonnes: float
    area_hectares: float
    average_height_meters: float
    change_7d_pct: float
    measurement_date: date
    confidence: float


class SolarFarmStatus(BaseModel):
    """Solar farm operational status."""
    farm_id: str
    location: Dict[str, float]
    capacity_mw: float
    panel_count: int
    operational_panels_pct: float
    soiling_index: float
    anomalies_detected: List[str]
    estimated_output_reduction_pct: float
    measurement_date: date


class PipelineMonitoring(BaseModel):
    """Pipeline integrity monitoring."""
    pipeline_id: str
    segment_id: str
    location: Dict[str, float]
    length_km: float
    anomalies: List[Dict]
    leak_probability: float
    vegetation_encroachment: bool
    third_party_activity: bool
    measurement_date: date


@router.get("/providers")
async def get_providers():
    """Get available satellite data providers."""
    return {
        "providers": [
            {
                "name": provider.value,
                "resolution_meters": info["resolution"],
                "revisit_days": info.get("revisit_days"),
                "all_weather": info.get("all_weather", False),
            }
            for provider, info in intelligence.providers.items()
        ]
    }


@router.get("/oil-storage/{tank_id}", response_model=StorageTankMeasurement)
async def measure_oil_tank(
    tank_id: str,
    lat: float = Query(...),
    lon: float = Query(...),
    provider: SatelliteProvider = SatelliteProvider.PLANET,
):
    """Measure oil storage tank level via satellite."""
    try:
        location = {"lat": lat, "lon": lon}
        result = intelligence.measure_oil_storage(tank_id, location, provider)
        return StorageTankMeasurement(**result)
    except Exception as e:
        logger.error(f"Error measuring tank: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coal-stockpile/{site_id}", response_model=CoalStockpileMeasurement)
async def measure_coal_stockpile(
    site_id: str,
    lat: float = Query(...),
    lon: float = Query(...),
    provider: SatelliteProvider = SatelliteProvider.SENTINEL,
):
    """Estimate coal stockpile volume via satellite."""
    try:
        location = {"lat": lat, "lon": lon}
        result = intelligence.measure_coal_stockpile(site_id, location, provider)
        return CoalStockpileMeasurement(**result)
    except Exception as e:
        logger.error(f"Error measuring stockpile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/solar-farm/{farm_id}", response_model=SolarFarmStatus)
async def analyze_solar_farm(
    farm_id: str,
    lat: float = Query(...),
    lon: float = Query(...),
    capacity_mw: float = Query(...),
    provider: SatelliteProvider = SatelliteProvider.PLANET,
):
    """Analyze solar farm operational status."""
    try:
        location = {"lat": lat, "lon": lon}
        result = intelligence.analyze_solar_farm(farm_id, location, capacity_mw, provider)
        return SolarFarmStatus(**result)
    except Exception as e:
        logger.error(f"Error analyzing solar farm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/{pipeline_id}/{segment_id}", response_model=PipelineMonitoring)
async def monitor_pipeline(
    pipeline_id: str,
    segment_id: str,
    lat: float = Query(...),
    lon: float = Query(...),
    provider: SatelliteProvider = SatelliteProvider.SAR,
):
    """Monitor pipeline integrity via satellite."""
    try:
        location = {"lat": lat, "lon": lon}
        result = intelligence.monitor_pipeline(pipeline_id, segment_id, location, provider)
        return PipelineMonitoring(**result)
    except Exception as e:
        logger.error(f"Error monitoring pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coverage")
async def get_coverage_stats():
    """Get satellite coverage statistics."""
    return {
        "oil_storage_tanks": 8547,
        "coal_stockpiles": 1234,
        "solar_farms": 3421,
        "wind_farms": 2156,
        "pipelines_km": 125000,
        "power_plants": 892,
        "daily_measurements": 15000,
        "historical_data_years": 8,
        "imagery_resolution_best": "3 meters",
    }
