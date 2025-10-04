"""
Risk Analytics Service
Value at Risk (VaR), Expected Shortfall, and portfolio risk aggregation.
"""
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import stats

# Use absolute imports to allow running as a module entrypoint
from var_calculator import VaRCalculator
from portfolio import PortfolioAggregator
from stress_testing import StressTestEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Risk Analytics Service",
    description="Portfolio risk metrics and stress testing",
    version="1.0.0",
)

# Initialize components
var_calculator = VaRCalculator()
portfolio_aggregator = PortfolioAggregator()
stress_engine = StressTestEngine()


class Position(BaseModel):
    """Trading position."""
    instrument_id: str
    quantity: float  # MW or contracts
    entry_price: Optional[float] = None


class VaRRequest(BaseModel):
    """VaR calculation request."""
    positions: List[Position]
    confidence_level: float = 0.95  # 95% confidence
    horizon_days: int = 1  # 1-day VaR
    method: str = "historical"  # historical, parametric, monte_carlo


class VaRResponse(BaseModel):
    """VaR calculation response."""
    var_value: float
    expected_shortfall: float
    confidence_level: float
    horizon_days: int
    method: str
    portfolio_value: float
    positions_count: int


class StressTestRequest(BaseModel):
    """Stress test request."""
    positions: List[Position]
    scenarios: List[Dict[str, Any]]


class StressTestResponse(BaseModel):
    """Stress test response."""
    scenario_results: List[Dict[str, Any]]
    worst_case_loss: float
    best_case_gain: float


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/risk/var", response_model=VaRResponse)
async def calculate_var(request: VaRRequest):
    """
    Calculate Value at Risk for a portfolio.
    
    Supports multiple methods:
    - Historical: Based on historical price movements
    - Parametric: Assumes normal distribution
    - Monte Carlo: Simulated price paths
    """
    logger.info(
        f"Calculating {request.confidence_level*100}% {request.horizon_days}-day VaR "
        f"for {len(request.positions)} positions using {request.method} method"
    )
    
    try:
        # Get historical prices for all instruments
        prices_data = await var_calculator.get_historical_prices(
            [p.instrument_id for p in request.positions],
            lookback_days=252,  # 1 year
        )
        
        # Build portfolio returns
        portfolio_returns = var_calculator.build_portfolio_returns(
            request.positions,
            prices_data,
        )
        
        # Calculate VaR based on method
        if request.method == "historical":
            var_value, es_value = var_calculator.historical_var(
                portfolio_returns,
                confidence_level=request.confidence_level,
                horizon_days=request.horizon_days,
            )
        elif request.method == "parametric":
            var_value, es_value = var_calculator.parametric_var(
                portfolio_returns,
                confidence_level=request.confidence_level,
                horizon_days=request.horizon_days,
            )
        elif request.method == "monte_carlo":
            var_value, es_value = var_calculator.monte_carlo_var(
                portfolio_returns,
                confidence_level=request.confidence_level,
                horizon_days=request.horizon_days,
                n_simulations=10000,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported VaR method: {request.method}"
            )
        
        # Calculate current portfolio value
        portfolio_value = var_calculator.calculate_portfolio_value(
            request.positions,
            prices_data,
        )
        
        return VaRResponse(
            var_value=var_value,
            expected_shortfall=es_value,
            confidence_level=request.confidence_level,
            horizon_days=request.horizon_days,
            method=request.method,
            portfolio_value=portfolio_value,
            positions_count=len(request.positions),
        )
        
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/risk/stress-test", response_model=StressTestResponse)
async def run_stress_test(request: StressTestRequest):
    """
    Run stress tests on portfolio.
    
    Scenarios include:
    - Price shocks (±20%, ±50%)
    - Volatility changes
    - Correlation breakdown
    - Historical crisis events
    """
    logger.info(
        f"Running {len(request.scenarios)} stress scenarios "
        f"on {len(request.positions)} positions"
    )
    
    try:
        results = []
        
        for scenario in request.scenarios:
            impact = await stress_engine.apply_scenario(
                request.positions,
                scenario,
            )
            
            results.append({
                "scenario_name": scenario.get("name", "Unnamed"),
                "pnl": impact["pnl"],
                "pnl_pct": impact["pnl_pct"],
                "details": impact.get("details", {}),
            })
        
        # Find worst and best cases
        worst_case = min(r["pnl"] for r in results)
        best_case = max(r["pnl"] for r in results)
        
        return StressTestResponse(
            scenario_results=results,
            worst_case_loss=worst_case,
            best_case_gain=best_case,
        )
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk/correlation-matrix")
async def get_correlation_matrix(
    instrument_ids: List[str],
    start_date: date,
    end_date: date,
):
    """
    Calculate correlation matrix for instruments.
    
    Useful for understanding portfolio diversification.
    """
    try:
        prices_data = await var_calculator.get_historical_prices(
            instrument_ids,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Calculate returns
        returns_df = prices_data.pct_change().dropna()
        
        # Correlation matrix
        corr_matrix = returns_df.corr()
        
        return {
            "instruments": instrument_ids,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "correlation_matrix": corr_matrix.to_dict(),
            "average_correlation": float(
                corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).mean().mean()
            ),
        }
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk/portfolio-summary")
async def get_portfolio_summary(positions: List[Position]):
    """Get comprehensive portfolio risk summary."""
    try:
        # Calculate various risk metrics
        var_95 = await calculate_var(
            VaRRequest(
                positions=positions,
                confidence_level=0.95,
                method="historical",
            )
        )
        
        var_99 = await calculate_var(
            VaRRequest(
                positions=positions,
                confidence_level=0.99,
                method="historical",
            )
        )
        
        # Portfolio statistics
        prices_data = await var_calculator.get_historical_prices(
            [p.instrument_id for p in positions],
            lookback_days=252,
        )
        
        portfolio_returns = var_calculator.build_portfolio_returns(
            positions,
            prices_data,
        )
        
        return {
            "portfolio_value": var_95.portfolio_value,
            "positions_count": len(positions),
            "var_95_1d": var_95.var_value,
            "var_99_1d": var_99.var_value,
            "expected_shortfall_95": var_95.expected_shortfall,
            "volatility_annual": float(portfolio_returns.std() * np.sqrt(252)),
            "sharpe_ratio": float(
                portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                if portfolio_returns.std() > 0 else 0
            ),
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
