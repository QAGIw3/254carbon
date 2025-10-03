"""
Battery Materials Service

Critical minerals and battery materials market intelligence including:
- Lithium (carbonate, hydroxide, spodumene)
- Cobalt, Nickel, Manganese, Graphite
- Rare earths (NdPr for magnets)
- Supply chain analytics and ESG scoring
"""
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Battery Materials Service",
    description="Critical minerals and battery materials intelligence",
    version="1.0.0",
)


class Material(str, Enum):
    LITHIUM_CARBONATE = "lithium_carbonate"
    LITHIUM_HYDROXIDE = "lithium_hydroxide"
    SPODUMENE = "spodumene"
    COBALT = "cobalt"
    NICKEL_SULFATE = "nickel_sulfate"
    NICKEL_CLASS1 = "nickel_class1"
    MANGANESE = "manganese"
    GRAPHITE_NATURAL = "graphite_natural"
    GRAPHITE_SYNTHETIC = "graphite_synthetic"
    NDPR = "ndpr"  # Neodymium-Praseodymium


class MaterialPrice(BaseModel):
    """Material price data."""
    material: Material
    date: date
    price: float
    currency: str
    unit: str
    exchange: str  # LME, CME, Chinese markets, spot
    contract_type: Optional[str] = None  # spot, 1M, 3M, etc.


class MineProduction(BaseModel):
    """Mine production data."""
    mine_id: str
    mine_name: str
    country: str
    operator: str
    material: Material
    annual_production_tonnes: float
    reserves_tonnes: float
    grade_pct: float
    status: str  # operating, development, exploration
    esg_score: float  # 0-100


class SupplyChainNode(BaseModel):
    """Supply chain node."""
    node_id: str
    node_type: str  # mine, processor, manufacturer, recycler
    location: str
    capacity_annual_tonnes: float
    material_input: Material
    material_output: Optional[Material] = None
    market_share_pct: float


class BatteryCostBreakdown(BaseModel):
    """Battery pack cost breakdown."""
    battery_chemistry: str  # NMC811, LFP, NCA, etc.
    cost_per_kwh: float
    materials_cost_pct: float
    manufacturing_cost_pct: float
    other_cost_pct: float
    material_breakdown: Dict[str, float]


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "battery-materials"}


@app.get("/api/v1/materials/prices", response_model=List[MaterialPrice])
async def get_material_prices(
    material: Optional[Material] = None,
    start_date: date = Query(...),
    end_date: date = Query(...),
    exchange: Optional[str] = None,
):
    """
    Get battery material prices.
    
    Sources: LME, CME, Chinese markets, spot indices
    """
    logger.info(f"Fetching prices for {material or 'all materials'}")
    
    # Base prices (USD/tonne or USD/kg depending on material)
    base_prices = {
        Material.LITHIUM_CARBONATE: {"price": 15000, "unit": "USD/tonne", "exchange": "China_Spot"},
        Material.LITHIUM_HYDROXIDE: {"price": 17000, "unit": "USD/tonne", "exchange": "China_Spot"},
        Material.SPODUMENE: {"price": 1200, "unit": "USD/tonne", "exchange": "Spot"},
        Material.COBALT: {"price": 32000, "unit": "USD/tonne", "exchange": "LME"},
        Material.NICKEL_SULFATE: {"price": 4500, "unit": "USD/tonne", "exchange": "China"},
        Material.NICKEL_CLASS1: {"price": 18500, "unit": "USD/tonne", "exchange": "LME"},
        Material.MANGANESE: {"price": 1800, "unit": "USD/tonne", "exchange": "China"},
        Material.GRAPHITE_NATURAL: {"price": 800, "unit": "USD/tonne", "exchange": "Spot"},
        Material.GRAPHITE_SYNTHETIC: {"price": 2500, "unit": "USD/tonne", "exchange": "Spot"},
        Material.NDPR: {"price": 65000, "unit": "USD/tonne", "exchange": "China"},
    }
    
    prices = []
    current_date = start_date
    
    while current_date <= end_date:
        for mat, config in base_prices.items():
            if material and mat != material:
                continue
            if exchange and config["exchange"] != exchange:
                continue
            
            # Add daily variation
            price = config["price"] * (1 + np.random.normal(0, 0.02))
            
            prices.append(MaterialPrice(
                material=mat,
                date=current_date,
                price=round(price, 2),
                currency="USD",
                unit=config["unit"],
                exchange=config["exchange"],
                contract_type="spot",
            ))
        
        current_date += timedelta(days=1)
        
        if len(prices) > 500:
            break
    
    return prices[:100]


@app.get("/api/v1/materials/mines", response_model=List[MineProduction])
async def get_mine_production(
    material: Optional[Material] = None,
    country: Optional[str] = None,
    min_production_tonnes: float = 0,
):
    """
    Get mine production data and reserves.
    
    Track global supply by mine and operator.
    """
    logger.info("Fetching mine production data")
    
    # Mock mine database
    mines = [
        {
            "mine_id": "MINE-001",
            "mine_name": "Greenbushes",
            "country": "Australia",
            "operator": "Tianqi/IGO",
            "material": Material.SPODUMENE,
            "annual_production": 1400000,  # tonnes
            "reserves": 50000000,
            "grade": 2.8,  # % Li2O
            "status": "operating",
            "esg_score": 75.0,
        },
        {
            "mine_id": "MINE-002",
            "mine_name": "Salar de Atacama",
            "country": "Chile",
            "operator": "SQM",
            "material": Material.LITHIUM_CARBONATE,
            "annual_production": 180000,
            "reserves": 8000000,
            "grade": 0.2,
            "status": "operating",
            "esg_score": 65.0,
        },
        {
            "mine_id": "MINE-003",
            "mine_name": "Kamoa-Kakula",
            "country": "DRC",
            "operator": "Ivanhoe Mines",
            "material": Material.COBALT,
            "annual_production": 15000,
            "reserves": 500000,
            "grade": 0.35,
            "status": "operating",
            "esg_score": 58.0,
        },
        {
            "mine_id": "MINE-004",
            "mine_name": "Norilsk",
            "country": "Russia",
            "operator": "Nornickel",
            "material": Material.NICKEL_CLASS1,
            "annual_production": 175000,
            "reserves": 6000000,
            "grade": 0.8,
            "status": "operating",
            "esg_score": 45.0,
        },
        {
            "mine_id": "MINE-005",
            "mine_name": "Mount Weld",
            "country": "Australia",
            "operator": "Lynas",
            "material": Material.NDPR,
            "annual_production": 6000,
            "reserves": 150000,
            "grade": 25.0,
            "status": "operating",
            "esg_score": 80.0,
        },
    ]
    
    # Filter
    filtered = []
    for mine in mines:
        if material and mine["material"] != material:
            continue
        if country and mine["country"] != country:
            continue
        if mine["annual_production"] < min_production_tonnes:
            continue
        
        filtered.append(MineProduction(
            mine_id=mine["mine_id"],
            mine_name=mine["mine_name"],
            country=mine["country"],
            operator=mine["operator"],
            material=mine["material"],
            annual_production_tonnes=mine["annual_production"],
            reserves_tonnes=mine["reserves"],
            grade_pct=mine["grade"],
            status=mine["status"],
            esg_score=mine["esg_score"],
        ))
    
    return filtered


@app.get("/api/v1/materials/supply-chain", response_model=List[SupplyChainNode])
async def get_supply_chain(
    material: Material,
):
    """
    Get supply chain map for material.
    
    Mine → Processing → Manufacturing → Recycling
    """
    logger.info(f"Fetching supply chain for {material}")
    
    # Mock supply chain
    if material == Material.LITHIUM_CARBONATE:
        chain = [
            SupplyChainNode(
                node_id="SC-001",
                node_type="mine",
                location="Australia",
                capacity_annual_tonnes=1400000,
                material_input=Material.SPODUMENE,
                material_output=Material.SPODUMENE,
                market_share_pct=25.0,
            ),
            SupplyChainNode(
                node_id="SC-002",
                node_type="processor",
                location="China",
                capacity_annual_tonnes=500000,
                material_input=Material.SPODUMENE,
                material_output=Material.LITHIUM_CARBONATE,
                market_share_pct=60.0,
            ),
            SupplyChainNode(
                node_id="SC-003",
                node_type="manufacturer",
                location="China",
                capacity_annual_tonnes=800000,
                material_input=Material.LITHIUM_CARBONATE,
                material_output=None,  # Final product
                market_share_pct=75.0,
            ),
            SupplyChainNode(
                node_id="SC-004",
                node_type="recycler",
                location="Europe",
                capacity_annual_tonnes=50000,
                material_input=Material.LITHIUM_CARBONATE,
                material_output=Material.LITHIUM_CARBONATE,
                market_share_pct=5.0,
            ),
        ]
    else:
        chain = []
    
    return chain


@app.get("/api/v1/materials/battery-cost", response_model=BatteryCostBreakdown)
async def calculate_battery_cost(
    chemistry: str = "NMC811",
    pack_size_kwh: float = 75.0,
):
    """
    Calculate battery pack cost breakdown.
    
    Chemistry: NMC811, NMC622, LFP, NCA, etc.
    """
    logger.info(f"Calculating cost for {chemistry} battery")
    
    # Cost breakdown by chemistry (2024 estimates, USD/kWh)
    chemistries = {
        "NMC811": {
            "total": 120.0,
            "materials_pct": 65.0,
            "lithium": 25.0,
            "nickel": 22.0,
            "cobalt": 8.0,
            "graphite": 5.0,
            "other": 5.0,
        },
        "NMC622": {
            "total": 130.0,
            "materials_pct": 68.0,
            "lithium": 20.0,
            "nickel": 20.0,
            "cobalt": 15.0,
            "graphite": 5.0,
            "other": 8.0,
        },
        "LFP": {
            "total": 95.0,
            "materials_pct": 55.0,
            "lithium": 18.0,
            "iron": 10.0,
            "phosphate": 8.0,
            "graphite": 12.0,
            "other": 7.0,
        },
        "NCA": {
            "total": 125.0,
            "materials_pct": 66.0,
            "lithium": 22.0,
            "nickel": 28.0,
            "cobalt": 3.0,
            "aluminum": 3.0,
            "graphite": 5.0,
            "other": 5.0,
        },
    }
    
    chem_data = chemistries.get(chemistry, chemistries["NMC811"])
    
    cost_per_kwh = chem_data["total"]
    materials_cost = cost_per_kwh * (chem_data["materials_pct"] / 100)
    manufacturing_cost = cost_per_kwh * 0.25
    other_cost = cost_per_kwh - materials_cost - manufacturing_cost
    
    # Material breakdown
    material_breakdown = {
        k: v for k, v in chem_data.items()
        if k not in ["total", "materials_pct"]
    }
    
    total_pack_cost = cost_per_kwh * pack_size_kwh
    
    return BatteryCostBreakdown(
        battery_chemistry=chemistry,
        cost_per_kwh=cost_per_kwh,
        materials_cost_pct=chem_data["materials_pct"],
        manufacturing_cost_pct=25.0,
        other_cost_pct=round(100 - chem_data["materials_pct"] - 25.0, 1),
        material_breakdown=material_breakdown,
    )


@app.get("/api/v1/materials/demand-forecast")
async def forecast_material_demand(
    material: Material,
    year: int = 2030,
):
    """
    Forecast material demand driven by EV and energy storage growth.
    
    Correlates with battery deployment projections.
    """
    logger.info(f"Forecasting {material} demand for {year}")
    
    # EV growth projections
    ev_sales_2024 = 14_000_000  # vehicles
    ev_cagr = 0.20  # 20% growth
    
    years_ahead = year - 2024
    ev_sales_forecast = ev_sales_2024 * ((1 + ev_cagr) ** years_ahead)
    
    # Battery size assumptions
    avg_battery_kwh = 65  # kWh per EV
    total_ev_batteries_gwh = (ev_sales_forecast * avg_battery_kwh) / 1000
    
    # Energy storage (grid-scale)
    storage_gwh_2024 = 50
    storage_cagr = 0.35
    storage_gwh_forecast = storage_gwh_2024 * ((1 + storage_cagr) ** years_ahead)
    
    total_batteries_gwh = total_ev_batteries_gwh + storage_gwh_forecast
    
    # Material intensity (kg per kWh)
    intensities = {
        Material.LITHIUM_CARBONATE: 0.65,  # kg/kWh
        Material.LITHIUM_HYDROXIDE: 0.70,
        Material.COBALT: 0.12,  # for NMC811
        Material.NICKEL_SULFATE: 0.55,
        Material.GRAPHITE_NATURAL: 0.90,
    }
    
    intensity = intensities.get(material, 0.5)
    
    # Calculate demand
    demand_tonnes = total_batteries_gwh * 1_000_000 * intensity / 1000
    
    # Current production capacity
    current_capacity = {
        Material.LITHIUM_CARBONATE: 800_000,
        Material.COBALT: 180_000,
        Material.NICKEL_SULFATE: 500_000,
    }
    
    capacity = current_capacity.get(material, 1_000_000)
    
    supply_deficit = demand_tonnes - capacity
    deficit_pct = (supply_deficit / capacity * 100) if capacity > 0 else 0
    
    return {
        "material": material,
        "year": year,
        "demand_tonnes": round(demand_tonnes, 0),
        "current_capacity_tonnes": capacity,
        "supply_deficit_tonnes": round(supply_deficit, 0),
        "deficit_pct": round(deficit_pct, 1),
        "drivers": {
            "ev_sales_million": round(ev_sales_forecast / 1_000_000, 1),
            "ev_batteries_gwh": round(total_ev_batteries_gwh, 0),
            "storage_gwh": round(storage_gwh_forecast, 0),
            "total_batteries_gwh": round(total_batteries_gwh, 0),
        },
    }


@app.get("/api/v1/materials/recycling")
async def get_recycling_economics(
    material: Material,
):
    """
    Calculate recycling economics.
    
    Critical for circular economy and supply security.
    """
    logger.info(f"Calculating recycling economics for {material}")
    
    # Mock recycling data
    recycling_data = {
        Material.LITHIUM_CARBONATE: {
            "recovery_rate": 0.90,  # 90% recovery
            "process_cost_usd_per_kg": 3.50,
            "energy_kwh_per_kg": 15,
            "virgin_price_usd_per_kg": 15.00,
            "recycled_price_usd_per_kg": 12.00,
        },
        Material.COBALT: {
            "recovery_rate": 0.95,
            "process_cost_usd_per_kg": 8.00,
            "energy_kwh_per_kg": 25,
            "virgin_price_usd_per_kg": 32.00,
            "recycled_price_usd_per_kg": 28.00,
        },
        Material.NICKEL_CLASS1: {
            "recovery_rate": 0.92,
            "process_cost_usd_per_kg": 5.00,
            "energy_kwh_per_kg": 20,
            "virgin_price_usd_per_kg": 18.50,
            "recycled_price_usd_per_kg": 16.00,
        },
    }
    
    data = recycling_data.get(material)
    if not data:
        raise HTTPException(status_code=404, detail=f"Recycling data not available for {material}")
    
    # Calculate economics
    virgin_cost = data["virgin_price_usd_per_kg"]
    recycled_cost = data["process_cost_usd_per_kg"]
    
    cost_savings = virgin_cost - recycled_cost
    savings_pct = (cost_savings / virgin_cost * 100)
    
    # Carbon emissions savings (kg CO2 per kg material)
    carbon_savings = {
        Material.LITHIUM_CARBONATE: 15.0,
        Material.COBALT: 25.0,
        Material.NICKEL_CLASS1: 20.0,
    }
    
    return {
        "material": material,
        "recovery_rate": data["recovery_rate"],
        "process_cost_usd_per_kg": data["process_cost_usd_per_kg"],
        "virgin_price_usd_per_kg": data["virgin_price_usd_per_kg"],
        "recycled_price_usd_per_kg": data["recycled_price_usd_per_kg"],
        "cost_savings_usd_per_kg": round(cost_savings, 2),
        "savings_pct": round(savings_pct, 1),
        "carbon_savings_kg_co2": carbon_savings.get(material, 10.0),
        "payback_vs_virgin": "immediate" if cost_savings > 0 else "not_competitive",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)

