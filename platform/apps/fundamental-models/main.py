"""
Fundamental Market Models Service

Long-term market modeling and capacity expansion:
- 10-year price projections
- Capacity expansion optimization
- Retirement schedule impacts
- Policy scenario analysis
- Technology cost curves
"""
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from enum import Enum

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fundamental Market Models",
    description="Long-term market modeling and capacity planning",
    version="1.0.0",
)


class FuelType(str, Enum):
    COAL = "coal"
    NATURAL_GAS = "natural_gas"
    NUCLEAR = "nuclear"
    HYDRO = "hydro"
    WIND = "wind"
    SOLAR = "solar"
    BATTERY = "battery"


class PolicyScenario(str, Enum):
    CURRENT_POLICY = "current_policy"
    NET_ZERO_2050 = "net_zero_2050"
    DELAYED_TRANSITION = "delayed_transition"
    AGGRESSIVE_RENEWABLES = "aggressive_renewables"


class LongTermProjection(BaseModel):
    """10-year price projection."""
    market: str
    scenario: PolicyScenario
    years: List[int]
    prices_usd_mwh: List[float]
    capacity_mix: Dict[str, List[float]]
    carbon_price_usd_tco2: List[float]


class CapacityExpansionPlan(BaseModel):
    """Optimal capacity expansion plan."""
    market: str
    scenario: PolicyScenario
    total_investment_billion: float
    annual_additions: Dict[int, Dict[str, float]]
    retirements: Dict[int, Dict[str, float]]
    cumulative_capacity: Dict[str, List[float]]


class FundamentalModeler:
    """Fundamental market modeling engine."""
    
    def __init__(self):
        self.technology_costs = self._load_technology_costs()
        self.carbon_scenarios = self._load_carbon_scenarios()
    
    def _load_technology_costs(self) -> Dict:
        """Load technology cost curves."""
        return {
            FuelType.WIND: {
                "capex_2024": 1400,  # USD/kW
                "learning_rate": 0.10,  # 10% cost reduction per doubling
                "opex_pct": 0.03,  # 3% of capex annually
            },
            FuelType.SOLAR: {
                "capex_2024": 1000,
                "learning_rate": 0.20,  # Faster learning
                "opex_pct": 0.02,
            },
            FuelType.BATTERY: {
                "capex_2024": 300,  # USD/kWh
                "learning_rate": 0.18,
                "opex_pct": 0.02,
            },
            FuelType.NATURAL_GAS: {
                "capex_2024": 800,
                "learning_rate": 0.02,  # Mature technology
                "opex_pct": 0.025,
            },
        }
    
    def _load_carbon_scenarios(self) -> Dict:
        """Load carbon price trajectories."""
        return {
            PolicyScenario.NET_ZERO_2050: [50, 75, 100, 150, 200, 250, 300, 350, 400, 450],
            PolicyScenario.CURRENT_POLICY: [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            PolicyScenario.DELAYED_TRANSITION: [25, 30, 35, 50, 80, 150, 250, 350, 450, 550],
            PolicyScenario.AGGRESSIVE_RENEWABLES: [60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        }
    
    def project_long_term_prices(
        self,
        market: str,
        scenario: PolicyScenario,
        horizon_years: int = 10
    ) -> Dict[str, Any]:
        """
        Project long-term wholesale power prices.
        
        Uses supply/demand equilibrium model.
        """
        logger.info(f"Projecting {horizon_years}-year prices for {market} under {scenario}")
        
        current_year = 2025
        years = list(range(current_year, current_year + horizon_years))
        
        # Base price (2025)
        base_price = 50.0  # USD/MWh
        
        # Carbon price trajectory
        carbon_prices = self.carbon_scenarios[scenario][:horizon_years]
        
        # Technology cost evolution
        solar_costs = []
        wind_costs = []
        solar_params = self.technology_costs[FuelType.SOLAR]
        wind_params = self.technology_costs[FuelType.WIND]
        
        for i in range(horizon_years):
            # Learning curve: cost * (cumulative_capacity / initial_capacity) ^ -learning_rate
            capacity_multiplier = (1 + i * 0.2) ** -solar_params["learning_rate"]
            solar_costs.append(solar_params["capex_2024"] * capacity_multiplier)
            
            capacity_multiplier = (1 + i * 0.15) ** -wind_params["learning_rate"]
            wind_costs.append(wind_params["capex_2024"] * capacity_multiplier)
        
        # Project prices
        prices = []
        capacity_mix = {"coal": [], "gas": [], "nuclear": [], "wind": [], "solar": [], "hydro": []}
        
        for i, year in enumerate(years):
            # Carbon cost impact (gas at 45% emissions, coal at 95%)
            carbon_impact_gas = carbon_prices[i] * 0.45
            carbon_impact_coal = carbon_prices[i] * 0.95
            
            # Gas becomes marginal generator as coal retires
            gas_share = min(0.30 + i * 0.03, 0.60)  # Growing gas share
            coal_share = max(0.40 - i * 0.05, 0.05)  # Declining coal
            renewable_share = min(0.20 + i * 0.04, 0.50)  # Growing renewables
            
            # Weighted marginal cost
            if coal_share > gas_share:
                # Coal on margin
                marginal_cost = base_price + carbon_impact_coal
            else:
                # Gas on margin
                marginal_cost = base_price + carbon_impact_gas
            
            # Renewable penetration discount
            renewable_discount = renewable_share * 0.15  # 15% price suppression at 100% renewables
            
            price = marginal_cost * (1 - renewable_discount)
            prices.append(round(price, 2))
            
            # Capacity mix evolution
            capacity_mix["coal"].append(round(coal_share * 100, 1))
            capacity_mix["gas"].append(round(gas_share * 100, 1))
            capacity_mix["nuclear"].append(15.0)  # Stable
            capacity_mix["wind"].append(round(renewable_share * 60, 1))
            capacity_mix["solar"].append(round(renewable_share * 40, 1))
            capacity_mix["hydro"].append(10.0)  # Stable
        
        return {
            "market": market,
            "scenario": scenario,
            "years": years,
            "prices_usd_mwh": prices,
            "capacity_mix": capacity_mix,
            "carbon_price_usd_tco2": carbon_prices,
        }
    
    def optimize_capacity_expansion(
        self,
        market: str,
        scenario: PolicyScenario,
        current_capacity: Dict[str, float],
        demand_growth_pct: float = 2.0
    ) -> Dict[str, Any]:
        """
        Optimize capacity expansion plan.
        
        Minimizes total system cost subject to reliability constraints.
        """
        logger.info(f"Optimizing capacity expansion for {market}")
        
        horizon_years = 10
        current_year = 2025
        
        # Annual demand growth
        base_demand = 50000  # MW
        annual_demand = [base_demand * (1 + demand_growth_pct/100) ** i for i in range(horizon_years)]
        
        # Carbon prices
        carbon_prices = self.carbon_scenarios[scenario][:horizon_years]
        
        # Optimization (simplified - use CVXPY in production)
        annual_additions = {}
        retirements = {}
        cumulative = {ft.value: [] for ft in FuelType}
        
        # Initialize with current capacity
        capacity = current_capacity.copy()
        
        for i, year in enumerate(range(current_year, current_year + horizon_years)):
            # Determine optimal additions based on carbon price
            carbon_price = carbon_prices[i]
            
            # Retirements (coal phaseout)
            coal_retirement = capacity.get("coal", 0) * 0.10  # 10% per year
            retirements[year] = {"coal": round(coal_retirement, 1)}
            capacity["coal"] = max(0, capacity.get("coal", 0) - coal_retirement)
            
            # Required new capacity
            demand = annual_demand[i]
            total_capacity = sum(capacity.values())
            reserve_margin = 1.15  # 15% reserve margin
            
            required_capacity = demand * reserve_margin - total_capacity
            
            if required_capacity > 0:
                # Optimize new builds
                if carbon_price > 100:
                    # High carbon price: favor renewables + storage
                    wind_add = required_capacity * 0.50
                    solar_add = required_capacity * 0.30
                    battery_add = required_capacity * 0.15
                    gas_add = required_capacity * 0.05
                else:
                    # Low carbon price: more gas
                    wind_add = required_capacity * 0.35
                    solar_add = required_capacity * 0.25
                    battery_add = required_capacity * 0.10
                    gas_add = required_capacity * 0.30
                
                annual_additions[year] = {
                    "wind": round(wind_add, 1),
                    "solar": round(solar_add, 1),
                    "battery": round(battery_add, 1),
                    "natural_gas": round(gas_add, 1),
                }
                
                capacity["wind"] = capacity.get("wind", 0) + wind_add
                capacity["solar"] = capacity.get("solar", 0) + solar_add
                capacity["battery"] = capacity.get("battery", 0) + battery_add
                capacity["natural_gas"] = capacity.get("natural_gas", 0) + gas_add
            else:
                annual_additions[year] = {}
            
            # Record cumulative capacity
            for fuel in FuelType:
                cumulative[fuel.value].append(round(capacity.get(fuel.value, 0), 1))
        
        # Calculate total investment
        total_investment = 0
        for year, additions in annual_additions.items():
            for fuel, mw in additions.items():
                tech_params = self.technology_costs.get(FuelType(fuel.replace("natural_", "").replace("_", "")))
                if tech_params:
                    capex = tech_params["capex_2024"] * mw * 1000  # Convert MW to kW
                    total_investment += capex / 1e9  # Convert to billions
        
        return {
            "market": market,
            "scenario": scenario,
            "total_investment_billion": round(total_investment, 2),
            "annual_additions": annual_additions,
            "retirements": retirements,
            "cumulative_capacity": cumulative,
        }


# Global modeler
modeler = FundamentalModeler()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "fundamental-models"}


@app.get("/api/v1/fundamentals/long-term-projection", response_model=LongTermProjection)
async def get_long_term_projection(
    market: str,
    scenario: PolicyScenario = PolicyScenario.CURRENT_POLICY,
    horizon_years: int = 10,
):
    """
    Get 10-year price projection based on fundamental modeling.
    
    Includes capacity mix evolution and carbon prices.
    """
    try:
        projection = modeler.project_long_term_prices(market, scenario, horizon_years)
        return LongTermProjection(**projection)
    except Exception as e:
        logger.error(f"Error generating projection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/fundamentals/capacity-expansion", response_model=CapacityExpansionPlan)
async def optimize_capacity_expansion(
    market: str,
    scenario: PolicyScenario = PolicyScenario.CURRENT_POLICY,
    demand_growth_pct: float = 2.0,
):
    """
    Optimize capacity expansion plan for next 10 years.
    
    Minimizes system cost while meeting demand and reliability constraints.
    """
    try:
        # Mock current capacity
        current_capacity = {
            "coal": 20000,
            "natural_gas": 15000,
            "nuclear": 10000,
            "wind": 8000,
            "solar": 5000,
            "hydro": 7000,
        }
        
        plan = modeler.optimize_capacity_expansion(market, scenario, current_capacity, demand_growth_pct)
        return CapacityExpansionPlan(**plan)
    except Exception as e:
        logger.error(f"Error optimizing expansion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/fundamentals/scenarios")
async def get_scenarios():
    """Get available policy scenarios."""
    return {
        "scenarios": [
            {"code": "current_policy", "name": "Current Policy", "description": "Continuation of existing policies"},
            {"code": "net_zero_2050", "name": "Net Zero 2050", "description": "Aggressive decarbonization to net zero by 2050"},
            {"code": "delayed_transition", "name": "Delayed Transition", "description": "Slow initial progress, rapid later transition"},
            {"code": "aggressive_renewables", "name": "Aggressive Renewables", "description": "Maximum renewable deployment"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8031)

