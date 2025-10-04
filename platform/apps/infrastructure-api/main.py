"""
Infrastructure REST API Service
--------------------------------

Provides REST endpoints for infrastructure data access including:
- Asset registry queries
- Time series data retrieval
- Aggregated statistics
- Data quality metrics
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import asyncpg
from clickhouse_driver import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Infrastructure Data API",
    description="REST API for energy infrastructure data",
    version="1.0.0",
)


# Pydantic Models

class AssetType(str, Enum):
    LNG_TERMINAL = "lng_terminal"
    POWER_PLANT = "power_plant"
    TRANSMISSION_LINE = "transmission_line"
    SUBSTATION = "substation"


class AssetStatus(str, Enum):
    OPERATIONAL = "operational"
    CONSTRUCTION = "construction"
    PLANNED = "planned"
    DECOMMISSIONED = "decommissioned"
    MOTHBALLED = "mothballed"


class InfrastructureAssetResponse(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: AssetType
    country: str
    region: Optional[str]
    latitude: float
    longitude: float
    status: AssetStatus
    operator: Optional[str]
    owner: Optional[str]
    commissioned_date: Optional[date]
    metadata: Dict[str, Any]


class PowerPlantResponse(InfrastructureAssetResponse):
    capacity_mw: float
    primary_fuel: str
    capacity_factor: Optional[float]
    annual_generation_gwh: Optional[float]


class LNGTerminalResponse(InfrastructureAssetResponse):
    storage_capacity_gwh: float
    regasification_capacity_gwh_d: Optional[float]
    num_tanks: Optional[int]


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float
    unit: str


class LNGInventoryResponse(BaseModel):
    terminal_id: str
    terminal_name: str
    country: str
    data: List[TimeSeriesPoint]


class PowerGenerationResponse(BaseModel):
    plant_id: str
    plant_name: str
    fuel_type: str
    data: List[TimeSeriesPoint]


class InfrastructureStatsResponse(BaseModel):
    date: date
    country: str
    asset_type: str
    total_capacity: float
    available_capacity: float
    num_assets: int
    avg_capacity_factor: Optional[float]


class DataQualityResponse(BaseModel):
    source: str
    last_update: datetime
    quality_score: float
    issues: List[Dict[str, Any]]


# Database connections

_pg_pool = None
_ch_client = None


async def get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = await asyncpg.create_pool(
            host="postgresql",
            port=5432,
            database="market_intelligence",
            user="postgres",
            password="postgres",
        )
    return _pg_pool


def get_ch_client():
    global _ch_client
    if _ch_client is None:
        _ch_client = Client(
            host="clickhouse",
            port=9000,
        )
    return _ch_client


# Infrastructure Asset Endpoints

@app.get("/api/v1/infrastructure/assets", response_model=List[InfrastructureAssetResponse])
async def get_infrastructure_assets(
    asset_type: Optional[AssetType] = None,
    country: Optional[str] = None,
    status: Optional[AssetStatus] = None,
    bbox: Optional[str] = Query(None, description="Bounding box: min_lat,min_lon,max_lat,max_lon"),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get infrastructure assets with filtering options."""
    
    pool = await get_pg_pool()
    
    query = """
        SELECT 
            asset_id, asset_name, asset_type, country, region,
            latitude, longitude, status, operator, owner,
            commissioned_date, metadata
        FROM pg.infrastructure_assets
        WHERE 1=1
    """
    params = []
    
    if asset_type:
        query += f" AND asset_type = ${len(params) + 1}"
        params.append(asset_type.value)
    
    if country:
        query += f" AND country = ${len(params) + 1}"
        params.append(country)
    
    if status:
        query += f" AND status = ${len(params) + 1}"
        params.append(status.value)
    
    if bbox:
        try:
            coords = [float(x) for x in bbox.split(",")]
            if len(coords) == 4:
                query += f"""
                    AND latitude >= ${len(params) + 1} 
                    AND longitude >= ${len(params) + 2}
                    AND latitude <= ${len(params) + 3}
                    AND longitude <= ${len(params) + 4}
                """
                params.extend(coords)
        except ValueError:
            raise HTTPException(400, "Invalid bbox format")
    
    query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
    params.extend([limit, offset])
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    return [
        InfrastructureAssetResponse(
            asset_id=r["asset_id"],
            asset_name=r["asset_name"],
            asset_type=r["asset_type"],
            country=r["country"],
            region=r["region"],
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            status=r["status"],
            operator=r["operator"],
            owner=r["owner"],
            commissioned_date=r["commissioned_date"],
            metadata=r["metadata"] or {},
        )
        for r in rows
    ]


@app.get("/api/v1/infrastructure/power-plants", response_model=List[PowerPlantResponse])
async def get_power_plants(
    country: Optional[str] = None,
    fuel_type: Optional[str] = None,
    min_capacity_mw: Optional[float] = None,
    operational_only: bool = True,
    limit: int = Query(100, le=1000),
):
    """Get power plants with detailed information."""
    
    pool = await get_pg_pool()
    
    query = """
        SELECT 
            a.*, p.capacity_mw, p.primary_fuel, p.capacity_factor,
            p.annual_generation_gwh
        FROM pg.infrastructure_assets a
        JOIN pg.power_plants p ON a.asset_id = p.asset_id
        WHERE a.asset_type = 'power_plant'
    """
    params = []
    
    if operational_only:
        query += " AND a.status = 'operational'"
    
    if country:
        query += f" AND a.country = ${len(params) + 1}"
        params.append(country)
    
    if fuel_type:
        query += f" AND p.primary_fuel = ${len(params) + 1}"
        params.append(fuel_type)
    
    if min_capacity_mw:
        query += f" AND p.capacity_mw >= ${len(params) + 1}"
        params.append(min_capacity_mw)
    
    query += f" ORDER BY p.capacity_mw DESC LIMIT ${len(params) + 1}"
    params.append(limit)
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    return [
        PowerPlantResponse(
            asset_id=r["asset_id"],
            asset_name=r["asset_name"],
            asset_type=r["asset_type"],
            country=r["country"],
            region=r["region"],
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            status=r["status"],
            operator=r["operator"],
            owner=r["owner"],
            commissioned_date=r["commissioned_date"],
            metadata=r["metadata"] or {},
            capacity_mw=float(r["capacity_mw"]),
            primary_fuel=r["primary_fuel"],
            capacity_factor=float(r["capacity_factor"]) if r["capacity_factor"] else None,
            annual_generation_gwh=float(r["annual_generation_gwh"]) if r["annual_generation_gwh"] else None,
        )
        for r in rows
    ]


@app.get("/api/v1/infrastructure/lng-terminals", response_model=List[LNGTerminalResponse])
async def get_lng_terminals(
    country: Optional[str] = None,
    min_capacity_gwh: Optional[float] = None,
    limit: int = Query(100, le=1000),
):
    """Get LNG terminals with storage details."""
    
    pool = await get_pg_pool()
    
    query = """
        SELECT 
            a.*, l.storage_capacity_gwh, l.regasification_capacity_gwh_d,
            l.num_tanks
        FROM pg.infrastructure_assets a
        JOIN pg.lng_terminals l ON a.asset_id = l.asset_id
        WHERE a.asset_type = 'lng_terminal'
    """
    params = []
    
    if country:
        query += f" AND a.country = ${len(params) + 1}"
        params.append(country)
    
    if min_capacity_gwh:
        query += f" AND l.storage_capacity_gwh >= ${len(params) + 1}"
        params.append(min_capacity_gwh)
    
    query += f" ORDER BY l.storage_capacity_gwh DESC LIMIT ${len(params) + 1}"
    params.append(limit)
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    return [
        LNGTerminalResponse(
            asset_id=r["asset_id"],
            asset_name=r["asset_name"],
            asset_type=r["asset_type"],
            country=r["country"],
            region=r["region"],
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            status=r["status"],
            operator=r["operator"],
            owner=r["owner"],
            commissioned_date=r["commissioned_date"],
            metadata=r["metadata"] or {},
            storage_capacity_gwh=float(r["storage_capacity_gwh"]),
            regasification_capacity_gwh_d=float(r["regasification_capacity_gwh_d"]) if r["regasification_capacity_gwh_d"] else None,
            num_tanks=r["num_tanks"],
        )
        for r in rows
    ]


# Time Series Data Endpoints

@app.get("/api/v1/infrastructure/lng-inventory/{terminal_id}", response_model=LNGInventoryResponse)
async def get_lng_inventory(
    terminal_id: str,
    start_date: date = Query(default=None, description="Start date (default: 30 days ago)"),
    end_date: date = Query(default=None, description="End date (default: today)"),
    metric: str = Query("inventory_gwh", description="Metric to retrieve"),
):
    """Get LNG terminal inventory time series."""
    
    if not start_date:
        start_date = date.today() - timedelta(days=30)
    if not end_date:
        end_date = date.today()
    
    ch_client = get_ch_client()
    
    # Map metric names to columns
    metric_map = {
        "inventory_gwh": "inventory_gwh",
        "inventory_mcm": "inventory_mcm",
        "fullness_pct": "fullness_pct",
        "send_out_gwh": "send_out_gwh",
    }
    
    if metric not in metric_map:
        raise HTTPException(400, f"Invalid metric. Choose from: {list(metric_map.keys())}")
    
    column = metric_map[metric]
    
    query = f"""
        SELECT 
            ts, terminal_name, country, {column}
        FROM market_intelligence.lng_terminal_data
        WHERE terminal_id = %(terminal_id)s
            AND ts >= %(start_date)s
            AND ts < %(end_date)s + INTERVAL 1 DAY
        ORDER BY ts
    """
    
    params = {
        "terminal_id": terminal_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    
    result = ch_client.execute(query, params)
    
    if not result:
        raise HTTPException(404, f"No data found for terminal {terminal_id}")
    
    # Get terminal info from first row
    terminal_name = result[0][1]
    country = result[0][2]
    
    # Extract time series data
    data = [
        TimeSeriesPoint(
            timestamp=row[0],
            value=row[3] if row[3] is not None else 0,
            unit="GWh" if "gwh" in metric else "mcm" if "mcm" in metric else "%",
        )
        for row in result
    ]
    
    return LNGInventoryResponse(
        terminal_id=terminal_id,
        terminal_name=terminal_name,
        country=country,
        data=data,
    )


@app.get("/api/v1/infrastructure/power-generation/{plant_id}", response_model=PowerGenerationResponse)
async def get_power_generation(
    plant_id: str,
    start_date: date = Query(default=None, description="Start date (default: 30 days ago)"),
    end_date: date = Query(default=None, description="End date (default: today)"),
    metric: str = Query("generation_mwh", description="Metric to retrieve"),
):
    """Get power plant generation time series."""
    
    if not start_date:
        start_date = date.today() - timedelta(days=30)
    if not end_date:
        end_date = date.today()
    
    ch_client = get_ch_client()
    
    # Map metric names to columns
    metric_map = {
        "generation_mwh": ("generation_mwh", "MWh"),
        "capacity_factor": ("capacity_factor", "%"),
        "availability_pct": ("availability_pct", "%"),
        "emissions_tco2": ("emissions_tco2", "tCO2"),
    }
    
    if metric not in metric_map:
        raise HTTPException(400, f"Invalid metric. Choose from: {list(metric_map.keys())}")
    
    column, unit = metric_map[metric]
    
    query = f"""
        SELECT 
            ts, plant_name, fuel_type, {column}
        FROM market_intelligence.power_plant_data
        WHERE plant_id = %(plant_id)s
            AND ts >= %(start_date)s
            AND ts < %(end_date)s + INTERVAL 1 DAY
        ORDER BY ts
    """
    
    params = {
        "plant_id": plant_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    
    result = ch_client.execute(query, params)
    
    if not result:
        raise HTTPException(404, f"No data found for plant {plant_id}")
    
    # Get plant info from first row
    plant_name = result[0][1]
    fuel_type = result[0][2]
    
    # Extract time series data
    data = [
        TimeSeriesPoint(
            timestamp=row[0],
            value=row[3] if row[3] is not None else 0,
            unit=unit,
        )
        for row in result
    ]
    
    return PowerGenerationResponse(
        plant_id=plant_id,
        plant_name=plant_name,
        fuel_type=fuel_type,
        data=data,
    )


# Aggregated Statistics

@app.get("/api/v1/infrastructure/stats/{country}", response_model=List[InfrastructureStatsResponse])
async def get_infrastructure_stats(
    country: str,
    asset_type: Optional[AssetType] = None,
    start_date: date = Query(default=None, description="Start date (default: 30 days ago)"),
    end_date: date = Query(default=None, description="End date (default: today)"),
):
    """Get aggregated infrastructure statistics by country."""
    
    if not start_date:
        start_date = date.today() - timedelta(days=30)
    if not end_date:
        end_date = date.today()
    
    ch_client = get_ch_client()
    
    query = """
        SELECT 
            date, country, asset_type,
            sum(total_capacity) as total_capacity,
            sum(available_capacity) as available_capacity,
            sum(num_assets) as num_assets,
            avg(avg_capacity_factor) as avg_capacity_factor
        FROM market_intelligence.infrastructure_daily_stats
        WHERE country = %(country)s
            AND date >= %(start_date)s
            AND date <= %(end_date)s
    """
    params = {
        "country": country,
        "start_date": start_date,
        "end_date": end_date,
    }
    
    if asset_type:
        query += " AND asset_type = %(asset_type)s"
        params["asset_type"] = asset_type.value
    
    query += " GROUP BY date, country, asset_type ORDER BY date DESC"
    
    result = ch_client.execute(query, params)
    
    return [
        InfrastructureStatsResponse(
            date=row[0],
            country=row[1],
            asset_type=row[2],
            total_capacity=row[3],
            available_capacity=row[4],
            num_assets=row[5],
            avg_capacity_factor=row[6],
        )
        for row in result
    ]


# Data Quality Endpoint

@app.get("/api/v1/infrastructure/data-quality", response_model=List[DataQualityResponse])
async def get_data_quality():
    """Get data quality metrics for infrastructure data sources."""
    
    pool = await get_pg_pool()
    
    query = """
        SELECT 
            s.source_id,
            c.last_event_time,
            c.last_successful_run,
            c.error_count,
            c.metadata
        FROM pg.source_registry s
        LEFT JOIN pg.connector_checkpoints c ON s.source_id = c.connector_id
        WHERE s.source_id IN ('alsi_lng_inventory', 'reexplorer_renewable', 
                              'wri_powerplants', 'gem_transmission')
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
    
    results = []
    for r in rows:
        # Calculate quality score based on error count and freshness
        error_count = r["error_count"] or 0
        last_update = r["last_event_time"] or r["last_successful_run"]
        
        quality_score = 100.0
        issues = []
        
        # Deduct points for errors
        if error_count > 0:
            quality_score -= min(error_count * 5, 50)
            issues.append({
                "type": "errors",
                "message": f"{error_count} recent errors",
                "severity": "warning" if error_count < 5 else "error",
            })
        
        # Check data freshness
        if last_update:
            age = datetime.now() - last_update.replace(tzinfo=None)
            if age > timedelta(days=7):
                quality_score -= 20
                issues.append({
                    "type": "staleness",
                    "message": f"Data is {age.days} days old",
                    "severity": "warning",
                })
        else:
            quality_score -= 30
            issues.append({
                "type": "no_data",
                "message": "No data received yet",
                "severity": "error",
            })
        
        results.append(DataQualityResponse(
            source=r["source_id"],
            last_update=last_update or datetime.min,
            quality_score=max(0, quality_score),
            issues=issues,
        ))
    
    return results


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "infrastructure-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8035)
