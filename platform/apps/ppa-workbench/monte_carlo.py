"""
Monte Carlo simulation engine for PPA analysis.
"""
import logging
from datetime import date
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """Monte Carlo simulation for price and generation paths."""
    
    def simulate_price_paths(
        self,
        start_date: date,
        end_date: date,
        n_simulations: int,
        initial_price: float,
        volatility: float,
        mean_reversion_speed: float = 0.3,
        long_term_mean: float = 45.0,
    ) -> np.ndarray:
        """
        Simulate correlated price paths using geometric Brownian motion
        with mean reversion.
        
        dP = κ(μ - P)dt + σP dW
        
        Args:
            start_date: Simulation start
            end_date: Simulation end
            n_simulations: Number of paths
            initial_price: Starting price
            volatility: Annual volatility
            mean_reversion_speed: Speed of mean reversion (κ)
            long_term_mean: Long-term average price (μ)
        
        Returns:
            Array of shape (n_simulations, n_periods)
        """
        n_years = (end_date - start_date).days / 365.25
        n_periods = int(n_years)  # Yearly periods
        dt = 1.0  # 1 year timestep
        
        # Initialize price paths
        paths = np.zeros((n_simulations, n_periods))
        paths[:, 0] = initial_price
        
        # Simulate paths
        for t in range(1, n_periods):
            # Random shocks
            dW = np.random.randn(n_simulations)
            
            # Mean reversion + diffusion
            drift = mean_reversion_speed * (long_term_mean - paths[:, t-1]) * dt
            diffusion = volatility * paths[:, t-1] * np.sqrt(dt) * dW
            
            paths[:, t] = paths[:, t-1] + drift + diffusion
            
            # No negative prices
            paths[:, t] = np.maximum(paths[:, t], 0.1)
        
        return paths
    
    def simulate_generation(
        self,
        capacity_mw: float,
        profile_type: str,
        n_years: int,
        n_simulations: int,
    ) -> np.ndarray:
        """
        Simulate generation profiles for renewable PPAs.
        
        Accounts for:
        - Inter-annual variability
        - Equipment degradation
        - Weather uncertainty
        
        Args:
            capacity_mw: Nameplate capacity
            profile_type: 'solar' or 'wind'
            n_years: Contract length
            n_simulations: Number of paths
        
        Returns:
            Array of annual MWh generation (n_simulations, n_years)
        """
        # Base capacity factors
        if profile_type == "solar":
            base_cf = 0.25  # 25% for solar
            cf_std = 0.03  # 3% inter-annual variability
        elif profile_type == "wind":
            base_cf = 0.35  # 35% for wind
            cf_std = 0.05  # 5% inter-annual variability
        else:
            base_cf = 0.90  # Baseload
            cf_std = 0.02
        
        # Annual degradation (0.5% per year for renewables)
        degradation_rate = 0.005 if profile_type in ["solar", "wind"] else 0
        
        # Simulate
        generation_paths = np.zeros((n_simulations, n_years))
        
        for sim in range(n_simulations):
            for year in range(n_years):
                # Random capacity factor
                cf = np.random.normal(base_cf, cf_std)
                cf = max(0.05, min(0.95, cf))  # Clamp
                
                # Apply degradation
                cf *= (1 - degradation_rate) ** year
                
                # Annual MWh
                annual_mwh = capacity_mw * 8760 * cf
                
                generation_paths[sim, year] = annual_mwh
        
        return generation_paths
    
    def simulate_correlated_variables(
        self,
        means: np.ndarray,
        cov_matrix: np.ndarray,
        n_simulations: int,
        n_periods: int,
    ) -> np.ndarray:
        """
        Simulate correlated random variables.
        
        Useful for multi-variate analysis (price, generation, basis).
        
        Returns:
            Array of shape (n_simulations, n_periods, n_variables)
        """
        n_vars = len(means)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)
        
        # Simulate
        simulations = np.zeros((n_simulations, n_periods, n_vars))
        
        for sim in range(n_simulations):
            for t in range(n_periods):
                # Uncorrelated random draws
                z = np.random.randn(n_vars)
                
                # Apply correlation
                correlated = means + L @ z
                
                simulations[sim, t, :] = correlated
        
        return simulations

