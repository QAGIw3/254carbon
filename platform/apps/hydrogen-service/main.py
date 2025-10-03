"""
Hydrogen Market Service

Comprehensive hydrogen economy tracking including:
- Green/blue/grey/pink/turquoise hydrogen pricing
- Electrolyzer capacity and project pipeline
- Ammonia, methanol, and SAF derivatives
- Transport and storage infrastructure
"""
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from enum import Enum

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hydrogen Market Service",
    description="Hydrogen economy market intelligence",
    version="1.0.0",
)


class HydrogenColor(str, Enum):
    GREEN = "green"  # Renewable electrolysis
    BLUE = "blue"  # Natural gas + CCS
    GREY = "grey"  # Natural gas, no CCS
    PINK = "pink"  # Nuclear-powered
    TURQUOISE = "turquoise"  # Methane pyrolysis


class HydrogenPrice(BaseModel):
    """Hydrogen price by type and region."""
    color: HydrogenColor
    region: str
    price_usd_per_kg: float
    date: date
    source: str
    currency: str = "USD"


class ElectrolyzerProject(BaseModel):
    """Electrolyzer project details."""
    project_id: str
    name: str
    location: str
    capacity_mw: float
    capacity_kg_per_day: float
    technology: str  # PEM, Alkaline, SOEC
    status: str  # planned, construction, operational
    expected_online: date
    capex_usd_million: float
    opex_usd_per_kg: float


class HydrogenDerivative(BaseModel):
    """Hydrogen derivative pricing."""
    product: str  # ammonia, methanol, SAF, steel
    price: float
    unit: str
    h2_content_pct: float
    conversion_efficiency: float


class TransportCost(BaseModel):
    """Hydrogen transport cost."""
    method: str  # pipeline, truck_compressed, truck_liquid, ship_ammonia
    distance_km: float
    cost_usd_per_kg: float
    energy_loss_pct: float


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "hydrogen"}


@app.get("/api/v1/hydrogen/prices", response_model=List[HydrogenPrice])
async def get_hydrogen_prices(
    color: Optional[HydrogenColor] = None,
    region: Optional[str] = None,
    start_date: date = Query(...),
    end_date: date = Query(...),
):
    """
    Get hydrogen prices by color and region.
    
    Regions: North America, Europe, Asia, Middle East
    """
    logger.info(f"Fetching H2 prices for {color or 'all'} in {region or 'all regions'}")
    
    # Mock price data - in production would fetch from market sources
    base_prices = {
        HydrogenColor.GREEN: {"NA": 6.50, "EU": 7.00, "ASIA": 8.00, "ME": 5.50},
        HydrogenColor.BLUE: {"NA": 3.50, "EU": 4.00, "ASIA": 4.50, "ME": 2.80},
        HydrogenColor.GREY: {"NA": 2.20, "EU": 2.80, "ASIA": 3.00, "ME": 1.80},
        HydrogenColor.PINK: {"NA": 5.50, "EU": 5.80, "ASIA": 6.50, "ME": 0.00},
        HydrogenColor.TURQUOISE: {"NA": 4.00, "EU": 4.50, "ASIA": 5.00, "ME": 3.50},
    }
    
    prices = []
    current_date = start_date
    
    while current_date <= end_date:
        for h2_color, regional_prices in base_prices.items():
            if color and h2_color != color:
                continue
            
            for reg, base_price in regional_prices.items():
                if region and reg != region:
                    continue
                
                if base_price == 0:  # Not available in region
                    continue
                
                # Add daily variation
                price = base_price + np.random.normal(0, 0.3)
                price = max(1.0, price)
                
                prices.append(HydrogenPrice(
                    color=h2_color,
                    region=reg,
                    price_usd_per_kg=round(price, 2),
                    date=current_date,
                    source="Market Data",
                ))
        
        current_date += timedelta(days=1)
    
    return prices[:100]  # Limit results


@app.get("/api/v1/hydrogen/projects", response_model=List[ElectrolyzerProject])
async def get_electrolyzer_projects(
    status: Optional[str] = None,
    region: Optional[str] = None,
    min_capacity_mw: float = 0,
):
    """
    Get electrolyzer project pipeline.
    
    Tracks global green hydrogen capacity buildout.
    """
    logger.info("Fetching electrolyzer projects")
    
    # Mock project database
    projects = [
        {
            "project_id": "PRJ-001",
            "name": "HyDeal Ambition",
            "location": "Spain",
            "capacity_mw": 9500,
            "technology": "PEM",
            "status": "planned",
            "expected_online": date(2030, 1, 1),
            "capex_usd_million": 15000,
        },
        {
            "project_id": "PRJ-002",
            "name": "NEOM Green Hydrogen",
            "location": "Saudi Arabia",
            "capacity_mw": 4000,
            "technology": "PEM",
            "status": "construction",
            "expected_online": date(2026, 12, 31),
            "capex_usd_million": 8500,
        },
        {
            "project_id": "PRJ-003",
            "name": "H2Giga Germany",
            "location": "Germany",
            "capacity_mw": 2000,
            "technology": "Alkaline",
            "status": "construction",
            "expected_online": date(2027, 6, 30),
            "capex_usd_million": 3000,
        },
        {
            "project_id": "PRJ-004",
            "name": "Port of Rotterdam H2",
            "location": "Netherlands",
            "capacity_mw": 1000,
            "technology": "PEM",
            "status": "operational",
            "expected_online": date(2024, 1, 1),
            "capex_usd_million": 1500,
        },
        {
            "project_id": "PRJ-005",
            "name": "Hydrogen Energy Supply Chain",
            "location": "Australia",
            "capacity_mw": 500,
            "technology": "Alkaline",
            "status": "operational",
            "expected_online": date(2023, 6, 1),
            "capex_usd_million": 800,
        },
    ]
    
    # Filter projects
    filtered = []
    for proj in projects:
        if status and proj["status"] != status:
            continue
        if region and region.lower() not in proj["location"].lower():
            continue
        if proj["capacity_mw"] < min_capacity_mw:
            continue
        
        # Calculate kg/day capacity (1 MW ≈ 480 kg H2/day at 50 kWh/kg efficiency)
        kg_per_day = proj["capacity_mw"] * 480
        
        # Calculate OPEX (simplified)
        opex_usd_per_kg = 1.5 + (hash(proj["project_id"]) % 10) / 10
        
        filtered.append(ElectrolyzerProject(
            project_id=proj["project_id"],
            name=proj["name"],
            location=proj["location"],
            capacity_mw=proj["capacity_mw"],
            capacity_kg_per_day=kg_per_day,
            technology=proj["technology"],
            status=proj["status"],
            expected_online=proj["expected_online"],
            capex_usd_million=proj["capex_usd_million"],
            opex_usd_per_kg=round(opex_usd_per_kg, 2),
        ))
    
    return filtered


@app.get("/api/v1/hydrogen/derivatives", response_model=List[HydrogenDerivative])
async def get_h2_derivatives():
    """
    Get hydrogen derivative prices.
    
    Tracks ammonia, methanol, SAF, and green steel.
    """
    logger.info("Fetching H2 derivatives")
    
    derivatives = [
        HydrogenDerivative(
            product="Green Ammonia",
            price=550.0,
            unit="USD/tonne",
            h2_content_pct=17.6,  # NH3 is 17.6% H2 by weight
            conversion_efficiency=0.85,
        ),
        HydrogenDerivative(
            product="Blue Ammonia",
            price=420.0,
            unit="USD/tonne",
            h2_content_pct=17.6,
            conversion_efficiency=0.85,
        ),
        HydrogenDerivative(
            product="Green Methanol",
            price=800.0,
            unit="USD/tonne",
            h2_content_pct=12.5,  # CH3OH
            conversion_efficiency=0.75,
        ),
        HydrogenDerivative(
            product="Sustainable Aviation Fuel (SAF)",
            price=2500.0,
            unit="USD/tonne",
            h2_content_pct=15.0,
            conversion_efficiency=0.65,
        ),
        HydrogenDerivative(
            product="Green Steel Premium",
            price=150.0,
            unit="USD/tonne_steel",
            h2_content_pct=2.5,  # H2 used per tonne steel
            conversion_efficiency=0.70,
        ),
    ]
    
    return derivatives


@app.get("/api/v1/hydrogen/transport", response_model=List[TransportCost])
async def get_transport_costs(
    distance_km: float = Query(..., ge=0),
    methods: List[str] = Query(None),
):
    """
    Calculate hydrogen transport costs.
    
    Methods: pipeline, truck_compressed, truck_liquid, ship_ammonia
    """
    logger.info(f"Calculating transport costs for {distance_km} km")
    
    # Base costs (USD/kg/1000km)
    transport_methods = {
        "pipeline": {"base_cost": 0.50, "energy_loss": 0.1},
        "truck_compressed": {"base_cost": 2.50, "energy_loss": 5.0},
        "truck_liquid": {"base_cost": 3.50, "energy_loss": 15.0},  # Liquefaction energy
        "ship_ammonia": {"base_cost": 1.20, "energy_loss": 8.0},
        "ship_lohc": {"base_cost": 2.00, "energy_loss": 12.0},  # Liquid Organic
    }
    
    results = []
    for method, params in transport_methods.items():
        if methods and method not in methods:
            continue
        
        # Calculate cost
        cost_per_kg = (params["base_cost"] * distance_km / 1000)
        
        # Add fixed costs
        if "truck" in method:
            cost_per_kg += 0.50  # Loading/unloading
        elif "ship" in method:
            cost_per_kg += 1.00  # Port operations
        
        results.append(TransportCost(
            method=method,
            distance_km=distance_km,
            cost_usd_per_kg=round(cost_per_kg, 2),
            energy_loss_pct=params["energy_loss"],
        ))
    
    return results


@app.get("/api/v1/hydrogen/economics")
async def calculate_h2_economics(
    production_method: HydrogenColor,
    electricity_price_usd_mwh: float = 40.0,
    natural_gas_price_usd_mmbtu: float = 4.0,
    capacity_factor: float = 0.50,
):
    """
    Calculate hydrogen production economics.
    
    Provides LCOH (Levelized Cost of Hydrogen) analysis.
    """
    logger.info(f"Calculating {production_method} H2 economics")
    
    # Electrolyzer costs (2024)
    capex_usd_per_kw = {
        HydrogenColor.GREEN: 1000,  # PEM electrolyzer
        HydrogenColor.PINK: 1100,   # Similar to green
        HydrogenColor.BLUE: 1500,   # SMR + CCS
        HydrogenColor.GREY: 800,    # SMR only
        HydrogenColor.TURQUOISE: 1200,  # Pyrolysis
    }
    
    # Energy consumption
    if production_method == HydrogenColor.GREEN or production_method == HydrogenColor.PINK:
        energy_kwh_per_kg = 50  # Modern PEM
        feedstock_cost = electricity_price_usd_mwh / 20  # USD/kg
    elif production_method == HydrogenColor.BLUE or production_method == HydrogenColor.GREY:
        energy_mmbtu_per_kg = 0.15  # Natural gas
        feedstock_cost = energy_mmbtu_per_kg * natural_gas_price_usd_mmbtu
    else:  # Turquoise
        energy_mmbtu_per_kg = 0.12
        feedstock_cost = energy_mmbtu_per_kg * natural_gas_price_usd_mmbtu
    
    # LCOH calculation
    capex = capex_usd_per_kw.get(production_method, 1000)
    annual_production_kg_per_kw = 8760 * capacity_factor / energy_kwh_per_kg if production_method in [HydrogenColor.GREEN, HydrogenColor.PINK] else 8760 * capacity_factor * 20
    
    capex_per_kg = (capex * 0.1) / annual_production_kg_per_kw  # 10% annualized
    opex_per_kg = 0.50  # O&M
    
    lcoh = capex_per_kg + opex_per_kg + feedstock_cost
    
    # Add CCS cost for blue
    if production_method == HydrogenColor.BLUE:
        lcoh += 0.80  # CCS cost
    
    return {
        "production_method": production_method,
        "lcoh_usd_per_kg": round(lcoh, 2),
        "capex_component": round(capex_per_kg, 2),
        "opex_component": round(opex_per_kg, 2),
        "feedstock_component": round(feedstock_cost, 2),
        "ccs_component": 0.80 if production_method == HydrogenColor.BLUE else 0.00,
        "annual_production_kg_per_kw": round(annual_production_kg_per_kw, 0),
        "capacity_factor": capacity_factor,
    }


@app.get("/api/v1/hydrogen/demand-forecast")
async def forecast_h2_demand(
    region: str,
    year: int = 2030,
):
    """
    Forecast hydrogen demand by sector and region.
    
    Sectors: transport, industry, power, buildings
    """
    logger.info(f"Forecasting H2 demand for {region} in {year}")
    
    # Mock demand forecast (million tonnes H2/year)
    base_demand_2024 = {
        "NA": {"transport": 0.5, "industry": 2.0, "power": 0.3, "buildings": 0.1},
        "EU": {"transport": 0.8, "industry": 3.5, "power": 0.5, "buildings": 0.2},
        "ASIA": {"transport": 1.2, "industry": 5.0, "power": 0.8, "buildings": 0.3},
        "ME": {"transport": 0.2, "industry": 1.0, "power": 0.1, "buildings": 0.0},
    }
    
    # Growth rates by sector
    growth_rates = {
        "transport": 0.25,  # 25% CAGR
        "industry": 0.15,
        "power": 0.30,
        "buildings": 0.20,
    }
    
    years_growth = year - 2024
    regional_demand = base_demand_2024.get(region, base_demand_2024["NA"])
    
    forecast = {}
    total_demand = 0
    
    for sector, base in regional_demand.items():
        demand = base * ((1 + growth_rates[sector]) ** years_growth)
        forecast[sector] = round(demand, 2)
        total_demand += demand
    
    return {
        "region": region,
        "year": year,
        "demand_by_sector_mt": forecast,
        "total_demand_mt": round(total_demand, 2),
        "growth_rate_cagr": {
            sector: f"{rate*100:.0f}%" for sector, rate in growth_rates.items()
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8019)

