"""
Monte Carlo simulation engine for scenario analysis and risk modeling.
Implements geometric Brownian motion and jump diffusion models.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations."""
    n_simulations: int = 10000
    n_steps: int = 252  # Trading days in a year
    dt: float = 1.0 / 252  # Time step (daily)
    risk_free_rate: float = 0.02  # 2% risk-free rate
    volatility: float = 0.3  # 30% volatility
    jump_intensity: float = 0.1  # Jump intensity (10% annual)
    jump_mean: float = 0.0  # Mean jump size
    jump_std: float = 0.2  # Standard deviation of jumps


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    price_paths: np.ndarray  # Shape: (n_simulations, n_steps + 1)
    final_prices: np.ndarray  # Shape: (n_simulations,)
    statistics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


class GeometricBrownianMotion:
    """Geometric Brownian Motion simulator for price paths."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate_paths(self, S0: float) -> np.ndarray:
        """
        Generate price paths using Geometric Brownian Motion.

        Args:
            S0: Initial price

        Returns:
            Array of shape (n_simulations, n_steps + 1) with price paths
        """
        n_sim, n_steps = self.config.n_simulations, self.config.n_steps
        dt = self.config.dt

        # Generate random normal variables
        Z = np.random.normal(0, 1, (n_sim, n_steps))

        # Initialize price paths
        paths = np.zeros((n_sim, n_steps + 1))
        paths[:, 0] = S0

        # Generate price paths
        for t in range(1, n_steps + 1):
            drift = (self.config.risk_free_rate - 0.5 * self.config.volatility**2) * dt
            diffusion = self.config.volatility * np.sqrt(dt) * Z[:, t-1]
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)

        return paths


class JumpDiffusion:
    """Jump diffusion model with Poisson jumps."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate_paths(self, S0: float) -> np.ndarray:
        """
        Generate price paths using Merton jump diffusion model.

        Args:
            S0: Initial price

        Returns:
            Array of shape (n_simulations, n_steps + 1) with price paths
        """
        n_sim, n_steps = self.config.n_simulations, self.config.n_steps
        dt = self.config.dt

        # Generate Brownian motion component
        Z = np.random.normal(0, 1, (n_sim, n_steps))

        # Generate jump times (Poisson process)
        jump_times = np.random.poisson(self.config.jump_intensity * dt, (n_sim, n_steps))

        # Initialize price paths
        paths = np.zeros((n_sim, n_steps + 1))
        paths[:, 0] = S0

        # Generate price paths
        for t in range(1, n_steps + 1):
            # Continuous component (GBM)
            drift = (self.config.risk_free_rate - 0.5 * self.config.volatility**2) * dt
            diffusion = self.config.volatility * np.sqrt(dt) * Z[:, t-1]

            # Jump component
            jump_sizes = np.random.normal(
                self.config.jump_mean,
                self.config.jump_std,
                n_sim
            )
            jump_component = jump_times[:, t-1] * jump_sizes

            # Combined process
            paths[:, t] = paths[:, t-1] * np.exp(
                drift + diffusion + jump_component
            )

        return paths


class MonteCarloEngine:
    """Main Monte Carlo simulation engine."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.gbm = GeometricBrownianMotion(self.config)
        self.jump_diffusion = JumpDiffusion(self.config)

    def run_simulation(
        self,
        S0: float,
        model: str = 'gbm',
        correlation_matrix: Optional[np.ndarray] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            S0: Initial price(s)
            model: 'gbm' for Geometric Brownian Motion, 'jump' for jump diffusion
            correlation_matrix: Correlation matrix for multi-asset simulation

        Returns:
            SimulationResult with price paths and statistics
        """
        logger.info(f"Running {model} simulation with {self.config.n_simulations} paths")

        if model.lower() == 'gbm':
            paths = self.gbm.generate_paths(S0)
        elif model.lower() == 'jump':
            paths = self.jump_diffusion.generate_paths(S0)
        else:
            raise ValueError(f"Unknown model: {model}")

        final_prices = paths[:, -1]
        statistics = self._calculate_statistics(final_prices)
        confidence_intervals = self._calculate_confidence_intervals(final_prices)

        return SimulationResult(
            price_paths=paths,
            final_prices=final_prices,
            statistics=statistics,
            confidence_intervals=confidence_intervals
        )

    def run_multi_asset_simulation(
        self,
        S0_vector: np.ndarray,
        correlation_matrix: np.ndarray,
        model: str = 'gbm'
    ) -> List[SimulationResult]:
        """
        Run correlated multi-asset Monte Carlo simulation.

        Args:
            S0_vector: Initial prices for each asset
            correlation_matrix: Correlation matrix between assets
            model: Simulation model to use

        Returns:
            List of SimulationResult objects, one per asset
        """
        n_assets = len(S0_vector)

        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError("Correlation matrix must match number of assets")

        # Cholesky decomposition for correlated random variables
        L = np.linalg.cholesky(correlation_matrix)

        results = []

        for i in range(n_assets):
            # Generate correlated random variables
            Z_uncorr = np.random.normal(0, 1, (self.config.n_simulations, self.config.n_steps))
            Z_corr = (L @ Z_uncorr.T).T

            # Run simulation for this asset
            if model.lower() == 'gbm':
                paths = self._generate_correlated_gbm_paths(S0_vector[i], Z_corr)
            else:
                raise ValueError(f"Multi-asset simulation not implemented for {model}")

            final_prices = paths[:, -1]
            statistics = self._calculate_statistics(final_prices)
            confidence_intervals = self._calculate_confidence_intervals(final_prices)

            results.append(SimulationResult(
                price_paths=paths,
                final_prices=final_prices,
                statistics=statistics,
                confidence_intervals=confidence_intervals
            ))

        return results

    def _generate_correlated_gbm_paths(self, S0: float, Z_corr: np.ndarray) -> np.ndarray:
        """Generate correlated GBM price paths."""
        n_sim, n_steps = Z_corr.shape
        dt = self.config.dt

        paths = np.zeros((n_sim, n_steps + 1))
        paths[:, 0] = S0

        for t in range(1, n_steps + 1):
            drift = (self.config.risk_free_rate - 0.5 * self.config.volatility**2) * dt
            diffusion = self.config.volatility * np.sqrt(dt) * Z_corr[:, t-1]
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)

        return paths

    def _calculate_statistics(self, final_prices: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistics for simulation results."""
        return {
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'std': np.std(final_prices),
            'min': np.min(final_prices),
            'max': np.max(final_prices),
            'skewness': stats.skew(final_prices),
            'kurtosis': stats.kurtosis(final_prices),
        }

    def _calculate_confidence_intervals(
        self,
        final_prices: np.ndarray,
        levels: List[float] = [0.05, 0.25, 0.75, 0.95]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for different probability levels."""
        intervals = {}
        for level in levels:
            alpha = 1 - level
            lower = np.percentile(final_prices, alpha * 100)
            upper = np.percentile(final_prices, level * 100)
            intervals[f'{int(level * 100)}%'] = (lower, upper)
        return intervals

    def calculate_var_cvar(
        self,
        result: SimulationResult,
        confidence_level: float = 0.95,
        initial_investment: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

        Args:
            result: Simulation result
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            initial_investment: Initial investment amount

        Returns:
            Dict with VaR and CVaR values
        """
        losses = initial_investment - result.final_prices
        losses = losses[losses > 0]  # Only consider losses

        if len(losses) == 0:
            return {'VaR': 0.0, 'CVaR': 0.0}

        # Value at Risk (VaR)
        var = np.percentile(losses, (1 - confidence_level) * 100)

        # Conditional VaR (Expected Shortfall)
        cvar = np.mean(losses[losses >= var])

        return {
            'VaR': var,
            'CVaR': cvar,
            'confidence_level': confidence_level
        }

    def option_pricing(
        self,
        S0: float,
        K: float,
        T: float,
        option_type: str = 'call',
        model: str = 'gbm'
    ) -> Dict[str, float]:
        """
        Price European options using Monte Carlo simulation.

        Args:
            S0: Current stock price
            K: Strike price
            T: Time to maturity (years)
            option_type: 'call' or 'put'
            model: Simulation model

        Returns:
            Dict with option price and Greeks
        """
        # Adjust time steps for option maturity
        n_steps = int(T / self.config.dt)
        dt = T / n_steps

        # Run simulation
        result = self.run_simulation(S0, model)
        final_prices = result.final_prices

        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - final_prices, 0)
        else:
            raise ValueError(f"Unknown option type: {option_type}")

        # Discount to present value
        option_price = np.mean(payoffs) * np.exp(-self.config.risk_free_rate * T)

        # Calculate delta (simplified)
        delta = np.mean((final_prices > K).astype(float)) if option_type == 'call' else -np.mean((final_prices < K).astype(float))

        return {
            'price': option_price,
            'delta': delta,
            'underlying_price': S0,
            'strike': K,
            'maturity': T,
            'option_type': option_type
        }
