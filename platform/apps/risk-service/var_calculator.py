"""
VaR calculation methods.
"""
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from clickhouse_driver import Client

logger = logging.getLogger(__name__)


class VaRCalculator:
    """Calculate Value at Risk using various methods."""
    
    def __init__(self):
        self.ch_client = Client(host="clickhouse", port=9000)
    
    async def get_historical_prices(
        self,
        instrument_ids: List[str],
        lookback_days: int = 252,
        start_date: date = None,
        end_date: date = None,
    ) -> pd.DataFrame:
        """Get historical prices for instruments."""
        if not start_date:
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=lookback_days)
        
        query = """
        SELECT 
            toDate(event_time) as date,
            instrument_id,
            avg(value) as price
        FROM ch.market_price_ticks
        WHERE instrument_id IN %(ids)s
          AND toDate(event_time) >= %(start_date)s
          AND toDate(event_time) <= %(end_date)s
          AND price_type = 'settle'
        GROUP BY date, instrument_id
        ORDER BY date, instrument_id
        """
        
        result = self.ch_client.execute(
            query,
            {
                "ids": tuple(instrument_ids),
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(result, columns=["date", "instrument_id", "price"])
        
        # Pivot to wide format
        prices_df = df.pivot(index="date", columns="instrument_id", values="price")
        prices_df.index = pd.to_datetime(prices_df.index)
        
        return prices_df
    
    def build_portfolio_returns(
        self,
        positions: List,
        prices_data: pd.DataFrame,
    ) -> pd.Series:
        """Build portfolio returns time series."""
        # Calculate individual returns
        returns = prices_data.pct_change().dropna()
        
        # Weight by position size
        weights = {}
        total_value = 0
        
        for pos in positions:
            if pos.instrument_id in prices_data.columns:
                latest_price = prices_data[pos.instrument_id].iloc[-1]
                value = pos.quantity * latest_price
                weights[pos.instrument_id] = value
                total_value += value
        
        # Normalize weights
        for inst_id in weights:
            weights[inst_id] /= total_value
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        
        for inst_id, weight in weights.items():
            if inst_id in returns.columns:
                portfolio_returns += returns[inst_id] * weight
        
        return portfolio_returns
    
    def historical_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> Tuple[float, float]:
        """
        Historical VaR: Use empirical distribution of returns.
        
        Returns:
            (VaR, Expected Shortfall)
        """
        # Scale returns to horizon
        scaled_returns = returns * np.sqrt(horizon_days)
        
        # VaR is the quantile
        var_percentile = 1 - confidence_level
        var_value = np.quantile(scaled_returns, var_percentile)
        
        # Expected Shortfall (CVaR): average of losses beyond VaR
        tail_losses = scaled_returns[scaled_returns <= var_value]
        expected_shortfall = tail_losses.mean() if len(tail_losses) > 0 else var_value
        
        return abs(var_value), abs(expected_shortfall)
    
    def parametric_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> Tuple[float, float]:
        """
        Parametric VaR: Assume normal distribution.
        
        Returns:
            (VaR, Expected Shortfall)
        """
        # Calculate mean and std
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Scale to horizon
        horizon_mean = mean_return * horizon_days
        horizon_std = std_return * np.sqrt(horizon_days)
        
        # VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var_value = -(horizon_mean + z_score * horizon_std)
        
        # Expected Shortfall for normal distribution
        # ES = -μ - σ * φ(Φ^-1(α)) / α
        phi_z = stats.norm.pdf(z_score)
        expected_shortfall = -(horizon_mean + horizon_std * phi_z / (1 - confidence_level))
        
        return abs(var_value), abs(expected_shortfall)
    
    def monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        n_simulations: int = 10000,
    ) -> Tuple[float, float]:
        """
        Monte Carlo VaR: Simulate future returns.
        
        Returns:
            (VaR, Expected Shortfall)
        """
        # Estimate parameters from historical data
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate simulated returns
        simulated_returns = np.random.normal(
            mean_return * horizon_days,
            std_return * np.sqrt(horizon_days),
            n_simulations,
        )
        
        # Calculate VaR and ES from simulations
        var_percentile = 1 - confidence_level
        var_value = np.quantile(simulated_returns, var_percentile)
        
        tail_losses = simulated_returns[simulated_returns <= var_value]
        expected_shortfall = tail_losses.mean()
        
        return abs(var_value), abs(expected_shortfall)
    
    def calculate_portfolio_value(
        self,
        positions: List,
        prices_data: pd.DataFrame,
    ) -> float:
        """Calculate current portfolio value."""
        total_value = 0.0
        
        for pos in positions:
            if pos.instrument_id in prices_data.columns:
                latest_price = prices_data[pos.instrument_id].iloc[-1]
                value = pos.quantity * latest_price
                total_value += value
        
        return total_value

