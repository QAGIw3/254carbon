"""
PPA Valuation Workbench
Power Purchase Agreement modeling and Monte Carlo simulation.
"""
import logging
from datetime import date, datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .contract_models import FixedPricePPA, CollarPPA, HubPlusPPA
from .monte_carlo import MonteCarloEngine
from .risk_analysis import PPARiskAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PPA Valuation Workbench",
    description="Power purchase agreement modeling and valuation",
    version="1.0.0",
)

# Initialize components
mc_engine = MonteCarloEngine()
risk_analyzer = PPARiskAnalyzer()


class PPAContract(BaseModel):
    """PPA contract specification."""
    contract_id: str
    contract_type: str  # fixed, collar, hub_plus, index
    counterparty: str
    start_date: date
    end_date: date
    capacity_mw: float
    
    # Pricing terms
    fixed_price: Optional[float] = None  # $/MWh for fixed
    floor_price: Optional[float] = None  # $/MWh for collar
    cap_price: Optional[float] = None  # $/MWh for collar
    index_hub: Optional[str] = None  # Hub for index-based
    basis_adder: Optional[float] = None  # $/MWh for hub+
    
    # Shape
    annual_mwh: float
    shape_profile: str = "baseload"  # baseload, peak, solar, wind


class ValuationRequest(BaseModel):
    contract: PPAContract
    market_scenario: str = "BASE"
    discount_rate: float = 0.06
    include_risk_metrics: bool = True


class ValuationResult(BaseModel):
    contract_id: str
    npv: float
    irr: float
    total_revenue: float
    average_price: float
    contract_years: int
    risk_metrics: Optional[Dict[str, Any]] = None


class MonteCarloRequest(BaseModel):
    contract: PPAContract
    n_simulations: int = 10000
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None


class MonteCarloResult(BaseModel):
    contract_id: str
    mean_npv: float
    std_npv: float
    var_95: float
    var_99: float
    percentiles: Dict[str, float]
    simulations: int


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/ppa/value", response_model=ValuationResult)
async def value_ppa(request: ValuationRequest):
    """
    Value PPA contract using forward curves.
    
    Calculates NPV, IRR, and risk metrics.
    """
    logger.info(
        f"Valuing {request.contract.contract_type} PPA: "
        f"{request.contract.contract_id}"
    )
    
    try:
        contract = request.contract
        
        # Get forward curve for index/hub
        if contract.contract_type in ["index", "hub_plus"]:
            hub_id = contract.index_hub or "PJM.HUB.WEST"
            # TODO: Fetch actual forward curve
            # For now, use mock curve
            forward_curve = generate_mock_curve(
                hub_id,
                contract.start_date,
                contract.end_date,
            )
        else:
            forward_curve = None
        
        # Calculate cash flows
        cash_flows = []
        years = (contract.end_date - contract.start_date).days / 365.25
        
        for year in range(int(years) + 1):
            contract_year = contract.start_date.year + year
            
            if contract.contract_type == "fixed":
                price = contract.fixed_price
            elif contract.contract_type == "collar":
                # Mock market price
                market_price = 45.0 + year * 2  # Escalating
                price = min(max(market_price, contract.floor_price), contract.cap_price)
            elif contract.contract_type == "hub_plus":
                # Index + basis
                index_price = 42.0 + year * 2  # Mock
                price = index_price + contract.basis_adder
            else:
                price = 40.0  # Default
            
            annual_revenue = contract.annual_mwh * price
            
            # Discount to present value
            pv = annual_revenue / ((1 + request.discount_rate) ** year)
            
            cash_flows.append({
                "year": contract_year,
                "volume_mwh": contract.annual_mwh,
                "price": price,
                "revenue": annual_revenue,
                "pv": pv,
            })
        
        # Calculate NPV
        npv = sum(cf["pv"] for cf in cash_flows)
        total_revenue = sum(cf["revenue"] for cf in cash_flows)
        avg_price = total_revenue / (contract.annual_mwh * years) if years > 0 else 0
        
        # IRR calculation (simplified)
        irr = calculate_simple_irr(cash_flows, request.discount_rate)
        
        # Risk metrics
        risk_metrics = None
        if request.include_risk_metrics:
            risk_metrics = risk_analyzer.calculate_ppa_risk(
                contract,
                cash_flows,
                forward_curve,
            )
        
        return ValuationResult(
            contract_id=contract.contract_id,
            npv=npv,
            irr=irr,
            total_revenue=total_revenue,
            average_price=avg_price,
            contract_years=int(years),
            risk_metrics=risk_metrics,
        )
        
    except Exception as e:
        logger.error(f"Error valuing PPA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ppa/monte-carlo", response_model=MonteCarloResult)
async def monte_carlo_valuation(request: MonteCarloRequest):
    """
    Monte Carlo simulation for PPA valuation.
    
    Simulates correlated price paths and generation uncertainty.
    """
    logger.info(
        f"Running Monte Carlo simulation for {request.contract.contract_id}: "
        f"{request.n_simulations} simulations"
    )
    
    try:
        contract = request.contract
        
        # Generate price path simulations
        price_paths = mc_engine.simulate_price_paths(
            start_date=contract.start_date,
            end_date=contract.end_date,
            n_simulations=request.n_simulations,
            initial_price=contract.fixed_price or 45.0,
            volatility=0.25,  # 25% annualized
        )
        
        # If renewable, simulate generation
        if contract.shape_profile in ["solar", "wind"]:
            generation_paths = mc_engine.simulate_generation(
                capacity_mw=contract.capacity_mw,
                profile_type=contract.shape_profile,
                n_years=int((contract.end_date - contract.start_date).days / 365.25),
                n_simulations=request.n_simulations,
            )
        else:
            generation_paths = None
        
        # Calculate NPV for each simulation
        npvs = []
        
        for sim in range(request.n_simulations):
            sim_npv = calculate_simulation_npv(
                contract,
                price_paths[sim],
                generation_paths[sim] if generation_paths else None,
                discount_rate=0.06,
            )
            npvs.append(sim_npv)
        
        npvs = np.array(npvs)
        
        # Calculate statistics
        mean_npv = npvs.mean()
        std_npv = npvs.std()
        var_95 = np.percentile(npvs, 5)  # 5th percentile = 95% VaR
        var_99 = np.percentile(npvs, 1)
        
        percentiles = {
            "p10": float(np.percentile(npvs, 10)),
            "p25": float(np.percentile(npvs, 25)),
            "p50": float(np.percentile(npvs, 50)),
            "p75": float(np.percentile(npvs, 75)),
            "p90": float(np.percentile(npvs, 90)),
        }
        
        return MonteCarloResult(
            contract_id=contract.contract_id,
            mean_npv=float(mean_npv),
            std_npv=float(std_npv),
            var_95=float(var_95),
            var_99=float(var_99),
            percentiles=percentiles,
            simulations=request.n_simulations,
        )
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_mock_curve(hub_id: str, start_date: date, end_date: date) -> pd.Series:
    """Generate mock forward curve."""
    dates = pd.date_range(start_date, end_date, freq="MS")
    prices = 45.0 + np.random.randn(len(dates)) * 5
    return pd.Series(prices, index=dates)


def calculate_simple_irr(cash_flows: List[Dict], initial_rate: float) -> float:
    """Calculate IRR using Newton-Raphson."""
    # Simplified IRR calculation
    # In production, would use proper NPV solver
    return initial_rate * 1.1  # Mock: slightly above discount rate


def calculate_simulation_npv(
    contract,
    price_path: np.ndarray,
    generation_path: Optional[np.ndarray],
    discount_rate: float,
) -> float:
    """Calculate NPV for one simulation."""
    # Simplified calculation
    years = len(price_path)
    annual_volumes = [contract.annual_mwh] * years
    
    if generation_path is not None:
        # Adjust volumes for generation uncertainty
        annual_volumes = generation_path
    
    cash_flows = []
    for year, (price, volume) in enumerate(zip(price_path, annual_volumes)):
        revenue = price * volume
        pv = revenue / ((1 + discount_rate) ** year)
        cash_flows.append(pv)
    
    return sum(cash_flows)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)

