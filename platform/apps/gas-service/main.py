"""
Natural Gas Service
Henry Hub futures, regional basis, storage, and pipeline data.
"""
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Natural Gas Service",
    description="Natural gas market data and analytics",
    version="1.0.0",
)


class HenryHubPrice(BaseModel):
    """Henry Hub price data."""
    date: date
    contract_month: str
    settlement_price: float
    volume: int
    open_interest: int


class BasisDifferential(BaseModel):
    """Regional basis differential."""
    hub_name: str
    date: date
    basis: float  # Differential to Henry Hub
    absolute_price: float


class StorageReport(BaseModel):
    """EIA storage report data."""
    report_date: date
    region: str
    inventory_bcf: float
    net_change_bcf: float
    year_ago_bcf: float
    five_year_avg_bcf: float


class PipelineFlow(BaseModel):
    """Pipeline flow and capacity data."""
    pipeline_name: str
    timestamp: datetime
    flow_mmbtu_day: float
    capacity_mmbtu_day: float
    utilization_pct: float


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/v1/gas/henry-hub", response_model=List[HenryHubPrice])
async def get_henry_hub_prices(
    start_date: date = Query(...),
    end_date: date = Query(...),
):
    """
    Get Henry Hub NYMEX futures prices.
    
    Data from CME/NYMEX natural gas futures (NG).
    """
    logger.info(f"Fetching Henry Hub prices from {start_date} to {end_date}")
    
    try:
        prices = []
        current_date = start_date
        
        while current_date <= end_date:
            # Generate contract months (next 12 months)
            for month_offset in range(12):
                contract_date = current_date + timedelta(days=30 * month_offset)
                contract_month = contract_date.strftime("%Y-%m")
                
                # Mock settlement price (typical range $2-$6/MMBtu)
                base_price = 3.50
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * contract_date.month / 12)
                settlement_price = base_price * seasonal_factor + np.random.normal(0, 0.2)
                
                prices.append(HenryHubPrice(
                    date=current_date,
                    contract_month=contract_month,
                    settlement_price=round(settlement_price, 3),
                    volume=int(50000 + np.random.normal(0, 10000)),
                    open_interest=int(200000 + np.random.normal(0, 50000)),
                ))
            
            current_date += timedelta(days=1)
            
            # Limit to avoid too many results
            if len(prices) > 1000:
                break
        
        return prices[:100]  # Return sample
        
    except Exception as e:
        logger.error(f"Error fetching Henry Hub prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/gas/basis", response_model=List[BasisDifferential])
async def get_basis_differentials(
    date_: date = Query(..., alias="date"),
    region: Optional[str] = None,
):
    """
    Get regional basis differentials to Henry Hub.
    
    Covers major trading hubs across North America.
    """
    logger.info(f"Fetching basis differentials for {date_}")
    
    try:
        # Major gas trading hubs
        hubs = [
            {"name": "Chicago City Gate", "typical_basis": 0.15},
            {"name": "Algonquin City Gate", "typical_basis": 0.50},
            {"name": "Transco Zone 6 NY", "typical_basis": 0.35},
            {"name": "SoCal Border", "typical_basis": -0.10},
            {"name": "PG&E City Gate", "typical_basis": 0.05},
            {"name": "AECO (Canada)", "typical_basis": -0.50},
            {"name": "Dawn Hub (Canada)", "typical_basis": -0.15},
            {"name": "Waha Hub (Permian)", "typical_basis": -1.20},
            {"name": "Dominion South", "typical_basis": -0.05},
            {"name": "Tennessee Zone 4", "typical_basis": 0.10},
        ]
        
        if region:
            hubs = [h for h in hubs if region.lower() in h["name"].lower()]
        
        # Henry Hub reference price (mock)
        henry_hub_price = 3.50
        
        basis_data = []
        for hub in hubs:
            # Add some random variation
            basis = hub["typical_basis"] + np.random.normal(0, 0.1)
            absolute_price = henry_hub_price + basis
            
            basis_data.append(BasisDifferential(
                hub_name=hub["name"],
                date=date_,
                basis=round(basis, 3),
                absolute_price=round(absolute_price, 3),
            ))
        
        return basis_data
        
    except Exception as e:
        logger.error(f"Error fetching basis differentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/gas/storage", response_model=List[StorageReport])
async def get_storage_data(
    start_date: date = Query(...),
    end_date: date = Query(...),
    region: str = Query("Lower 48"),
):
    """
    Get EIA natural gas storage data.
    
    Weekly storage reports by region.
    """
    logger.info(f"Fetching storage data for {region} from {start_date} to {end_date}")
    
    try:
        # EIA regions
        regions_map = {
            "Lower 48": {"capacity": 4000, "seasonal_low": 1500, "seasonal_high": 3800},
            "East": {"capacity": 1500, "seasonal_low": 600, "seasonal_high": 1400},
            "Midwest": {"capacity": 1000, "seasonal_low": 400, "seasonal_high": 950},
            "Mountain": {"capacity": 200, "seasonal_low": 80, "seasonal_high": 190},
            "Pacific": {"capacity": 300, "seasonal_low": 120, "seasonal_high": 280},
            "South Central": {"capacity": 1000, "seasonal_low": 300, "seasonal_high": 980},
        }
        
        region_data = regions_map.get(region, regions_map["Lower 48"])
        
        reports = []
        current_date = start_date
        
        while current_date <= end_date:
            # Weekly reports (Thursdays)
            if current_date.weekday() == 3:  # Thursday
                # Seasonal pattern
                day_of_year = current_date.timetuple().tm_yday
                seasonal_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - 365/4) / 365)
                
                inventory = region_data["seasonal_low"] + (
                    region_data["seasonal_high"] - region_data["seasonal_low"]
                ) * seasonal_factor
                
                # Weekly change (typically -100 to +100 Bcf)
                net_change = np.random.normal(0, 50)
                
                # Year ago (slightly different)
                year_ago = inventory + np.random.normal(0, 200)
                
                # 5-year average
                five_year_avg = (region_data["seasonal_low"] + region_data["seasonal_high"]) / 2
                
                reports.append(StorageReport(
                    report_date=current_date,
                    region=region,
                    inventory_bcf=round(inventory, 1),
                    net_change_bcf=round(net_change, 1),
                    year_ago_bcf=round(year_ago, 1),
                    five_year_avg_bcf=round(five_year_avg, 1),
                ))
            
            current_date += timedelta(days=1)
        
        return reports
        
    except Exception as e:
        logger.error(f"Error fetching storage data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/gas/pipelines", response_model=List[PipelineFlow])
async def get_pipeline_flows(
    pipeline: Optional[str] = None,
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
):
    """
    Get pipeline flow and capacity data.
    
    Real-time and historical pipeline utilization.
    """
    logger.info(f"Fetching pipeline flows from {start_time} to {end_time}")
    
    try:
        # Major interstate pipelines
        pipelines = [
            {"name": "Transco", "capacity": 15000},
            {"name": "Texas Eastern", "capacity": 9000},
            {"name": "ANR", "capacity": 5000},
            {"name": "Tennessee Gas", "capacity": 7000},
            {"name": "Southern Natural Gas", "capacity": 4000},
            {"name": "Columbia Gas", "capacity": 6000},
            {"name": "El Paso Natural Gas", "capacity": 5500},
            {"name": "Kern River", "capacity": 2000},
        ]
        
        if pipeline:
            pipelines = [p for p in pipelines if pipeline.lower() in p["name"].lower()]
        
        flows = []
        current_time = start_time
        
        while current_time <= end_time:
            for pipe in pipelines:
                # Mock flow (60-95% of capacity typically)
                utilization = 60 + np.random.normal(20, 10)
                utilization = max(40, min(100, utilization))
                
                flow = pipe["capacity"] * (utilization / 100)
                
                flows.append(PipelineFlow(
                    pipeline_name=pipe["name"],
                    timestamp=current_time,
                    flow_mmbtu_day=round(flow, 2),
                    capacity_mmbtu_day=float(pipe["capacity"]),
                    utilization_pct=round(utilization, 2),
                ))
            
            current_time += timedelta(hours=1)
            
            # Limit results
            if len(flows) > 500:
                break
        
        return flows[:100]  # Return sample
        
    except Exception as e:
        logger.error(f"Error fetching pipeline flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/gas/lng")
async def get_lng_data(
    facility: Optional[str] = None,
):
    """
    Get LNG export/import facility data.
    
    Track liquefaction and regasification capacity and utilization.
    """
    logger.info("Fetching LNG facility data")
    
    try:
        # Major US LNG facilities
        facilities = [
            {
                "name": "Sabine Pass",
                "location": "Louisiana",
                "type": "export",
                "capacity_bcfd": 6.0,
                "current_utilization": 85,
            },
            {
                "name": "Corpus Christi",
                "location": "Texas",
                "type": "export",
                "capacity_bcfd": 3.5,
                "current_utilization": 78,
            },
            {
                "name": "Cameron LNG",
                "location": "Louisiana",
                "type": "export",
                "capacity_bcfd": 3.0,
                "current_utilization": 90,
            },
            {
                "name": "Freeport LNG",
                "location": "Texas",
                "type": "export",
                "capacity_bcfd": 2.1,
                "current_utilization": 72,
            },
            {
                "name": "Cove Point",
                "location": "Maryland",
                "type": "export",
                "capacity_bcfd": 0.8,
                "current_utilization": 95,
            },
        ]
        
        if facility:
            facilities = [f for f in facilities if facility.lower() in f["name"].lower()]
        
        return {"facilities": facilities, "total_export_capacity_bcfd": 15.4}
        
    except Exception as e:
        logger.error(f"Error fetching LNG data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)

