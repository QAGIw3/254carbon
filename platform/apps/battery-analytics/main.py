"""
Battery Storage Analytics Service
Revenue optimization, degradation modeling, and dispatch optimization.
"""
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.optimize import linprog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Battery Storage Analytics",
    description="Battery revenue optimization and analytics",
    version="1.0.0",
)


class BatterySpec(BaseModel):
    """Battery system specification."""
    battery_id: str
    capacity_mwh: float
    power_rating_mw: float
    efficiency: float = 0.85  # Round-trip efficiency
    degradation_rate: float = 0.02  # 2% per year
    initial_soc: float = 0.5  # State of charge (0-1)


class MarketPrices(BaseModel):
    """Market price forecast."""
    intervals: List[Dict[str, Any]]  # [{time, energy_price, as_price}]


class OptimizationRequest(BaseModel):
    battery: BatterySpec
    prices: MarketPrices
    optimization_horizon_hours: int = 24


class DispatchSchedule(BaseModel):
    """Optimal dispatch schedule."""
    interval_ending: datetime
    charge_discharge_mw: float  # Positive = discharge, negative = charge
    soc: float
    revenue: float
    action: str  # CHARGE, DISCHARGE, IDLE


class OptimizationResult(BaseModel):
    battery_id: str
    total_revenue: float
    energy_arbitrage_revenue: float
    ancillary_service_revenue: float
    degradation_cost: float
    net_revenue: float
    schedule: List[DispatchSchedule]


class DegradationAnalysis(BaseModel):
    battery_id: str
    current_capacity_pct: float
    cycles_completed: int
    estimated_remaining_life_years: float
    replacement_year: int


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/battery/optimize", response_model=OptimizationResult)
async def optimize_dispatch(request: OptimizationRequest):
    """
    Optimize battery dispatch for maximum revenue.
    
    Considers:
    - Energy arbitrage (charge low, discharge high)
    - Ancillary service opportunities
    - Degradation costs
    - SOC constraints
    - Power rating limits
    """
    logger.info(
        f"Optimizing dispatch for {request.battery.battery_id}, "
        f"horizon={request.optimization_horizon_hours}h"
    )
    
    try:
        battery = request.battery
        n_intervals = len(request.prices.intervals)
        
        # Decision variables: charge/discharge power for each interval
        # Positive = discharge, Negative = charge
        
        # Build optimization problem
        # Objective: maximize revenue - degradation cost
        
        # Energy prices for each interval
        energy_prices = np.array([
            p.get("energy_price", 0) for p in request.prices.intervals
        ])
        
        # Ancillary service prices (if available)
        as_prices = np.array([
            p.get("as_price", 0) for p in request.prices.intervals
        ])
        
        # Simple linear programming for energy arbitrage
        # In production, would use MILP with degradation and AS
        
        optimal_schedule = []
        soc = battery.initial_soc
        total_revenue = 0.0
        energy_revenue = 0.0
        as_revenue = 0.0
        
        for i, interval_data in enumerate(request.prices.intervals):
            interval_time = datetime.fromisoformat(interval_data["time"])
            energy_price = interval_data.get("energy_price", 0)
            as_price = interval_data.get("as_price", 0)
            
            # Simple strategy: discharge when price > average, charge when below
            avg_price = energy_prices.mean()
            
            if energy_price > avg_price * 1.1 and soc > 0.2:
                # Discharge
                power = min(
                    battery.power_rating_mw,
                    (soc - 0.1) * battery.capacity_mwh * 4,  # 15-min to hourly
                )
                action = "DISCHARGE"
                revenue = power * energy_price * 0.25  # 15-min interval
                soc -= (power * 0.25) / battery.capacity_mwh
                energy_revenue += revenue
                
            elif energy_price < avg_price * 0.9 and soc < 0.9:
                # Charge
                power = -min(
                    battery.power_rating_mw,
                    (0.9 - soc) * battery.capacity_mwh * 4,
                )
                action = "CHARGE"
                revenue = power * energy_price * 0.25  # Negative (cost)
                soc += (-power * 0.25 * battery.efficiency) / battery.capacity_mwh
                energy_revenue += revenue
                
            else:
                # Idle (potentially provide ancillary services)
                power = 0
                action = "IDLE"
                revenue = 0
                
                # Could provide regulation
                if as_price > 5.0:
                    as_capacity = battery.power_rating_mw * 0.5
                    as_revenue += as_capacity * as_price * 0.25
            
            total_revenue += revenue
            
            # Clamp SOC
            soc = max(0.0, min(1.0, soc))
            
            optimal_schedule.append(DispatchSchedule(
                interval_ending=interval_time,
                charge_discharge_mw=power,
                soc=soc,
                revenue=revenue,
                action=action,
            ))
        
        # Calculate degradation cost
        cycles = sum(abs(s.charge_discharge_mw) for s in optimal_schedule) / battery.capacity_mwh
        degradation_cost = cycles * battery.capacity_mwh * 50.0  # $50/MWh-cycle
        
        net_revenue = total_revenue + as_revenue - degradation_cost
        
        return OptimizationResult(
            battery_id=battery.battery_id,
            total_revenue=total_revenue,
            energy_arbitrage_revenue=energy_revenue,
            ancillary_service_revenue=as_revenue,
            degradation_cost=degradation_cost,
            net_revenue=net_revenue,
            schedule=optimal_schedule,
        )
        
    except Exception as e:
        logger.error(f"Error optimizing battery dispatch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/battery/degradation", response_model=DegradationAnalysis)
async def analyze_degradation(
    battery_id: str,
    installation_date: date,
    capacity_mwh: float,
    cycles_per_day: float = 1.0,
):
    """
    Analyze battery degradation and estimate remaining life.
    
    Uses:
    - Calendar aging
    - Cycle aging
    - Depth of discharge impact
    """
    logger.info(f"Analyzing degradation for {battery_id}")
    
    try:
        # Calculate age
        age_years = (date.today() - installation_date).days / 365.25
        
        # Calendar degradation: ~2% per year
        calendar_degradation = age_years * 0.02
        
        # Cycle degradation
        total_cycles = cycles_per_day * age_years * 365
        cycle_degradation = total_cycles / 5000  # Assume 5000 cycle life
        
        # Total degradation
        total_degradation = min(calendar_degradation + cycle_degradation, 0.8)
        
        # Current capacity
        current_capacity_pct = (1 - total_degradation) * 100
        
        # Remaining life (until 80% capacity)
        remaining_degradation = 0.20 - total_degradation
        if remaining_degradation > 0:
            remaining_life_years = remaining_degradation / 0.02  # At 2% per year
        else:
            remaining_life_years = 0
        
        replacement_year = installation_date.year + int(age_years + remaining_life_years)
        
        return DegradationAnalysis(
            battery_id=battery_id,
            current_capacity_pct=current_capacity_pct,
            cycles_completed=int(total_cycles),
            estimated_remaining_life_years=remaining_life_years,
            replacement_year=replacement_year,
        )
        
    except Exception as e:
        logger.error(f"Error analyzing degradation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/battery/revenue-stack")
async def calculate_revenue_stack(
    battery: BatterySpec,
    market_prices: MarketPrices,
):
    """
    Calculate stacked revenue from multiple services.
    
    Revenue sources:
    1. Energy arbitrage (charge/discharge)
    2. Frequency regulation
    3. Spinning reserve
    4. Capacity payments
    """
    try:
        # Optimize for each revenue stream
        
        # 1. Energy arbitrage (primary)
        arbitrage_result = await optimize_dispatch(
            OptimizationRequest(
                battery=battery,
                prices=market_prices,
            )
        )
        
        # 2. Frequency regulation (mock)
        # Can provide regulation when idle
        reg_capacity_mw = battery.power_rating_mw * 0.5
        reg_price_avg = 12.0  # $/MW
        reg_hours = 8760 * 0.5  # 50% availability
        annual_reg_revenue = reg_capacity_mw * reg_price_avg * reg_hours
        
        # 3. Capacity market (mock)
        capacity_price = 150.0  # $/MW-day
        annual_capacity_revenue = battery.power_rating_mw * capacity_price * 365
        
        # Total annual revenue estimate
        annual_arbitrage = arbitrage_result.net_revenue * 365  # Scale daily to annual
        total_annual_revenue = (
            annual_arbitrage +
            annual_reg_revenue +
            annual_capacity_revenue
        )
        
        return {
            "battery_id": battery.battery_id,
            "revenue_stack": {
                "energy_arbitrage": annual_arbitrage,
                "frequency_regulation": annual_reg_revenue,
                "capacity_payments": annual_capacity_revenue,
                "total": total_annual_revenue,
            },
            "revenue_per_mw": total_annual_revenue / battery.power_rating_mw,
            "revenue_per_mwh": total_annual_revenue / battery.capacity_mwh,
            "payback_period_years": 1000000 / total_annual_revenue if total_annual_revenue > 0 else 999,  # Mock capex $1M/MWh
        }
        
    except Exception as e:
        logger.error(f"Error calculating revenue stack: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)

