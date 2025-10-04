"""
Storage Arbitrage Models

Advanced models for natural gas storage optimization and arbitrage:
- Seasonal storage value calculation
- Injection/withdrawal optimization
- Time spread arbitrage strategies
- Storage capacity valuation
- Regional basis optimization
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StorageArbitrageModel:
    """
    Natural gas storage arbitrage and optimization model.

    Features:
    - Seasonal storage value calculation
    - Optimal injection/withdrawal timing
    - Time spread arbitrage strategies
    - Storage capacity valuation
    """

    def __init__(self):
        self.storage_costs = {
            'injection_cost': 0.02,  # $/MMBtu injection cost
            'withdrawal_cost': 0.02,  # $/MMBtu withdrawal cost
            'storage_cost': 0.001,   # $/MMBtu/month storage cost
            'fuel_cost': 0.005       # $/MMBtu fuel cost for compression
        }

        self.seasonal_patterns = self._calculate_seasonal_patterns()

    def _calculate_seasonal_patterns(self) -> Dict[str, pd.Series]:
        """Calculate historical seasonal patterns for storage optimization."""

        # Mock seasonal data - in production, use historical price data
        dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')

        # Generate seasonal price patterns
        seasonal_prices = pd.Series(index=dates)

        for date in dates:
            # Base price with seasonal variation
            base_price = 3.5  # $/MMBtu

            # Winter premium (heating season)
            if date.month in [12, 1, 2, 3]:
                seasonal_multiplier = 1.3
            # Summer discount (injection season)
            elif date.month in [6, 7, 8]:
                seasonal_multiplier = 0.8
            else:
                seasonal_multiplier = 1.0

            # Weekend discount
            if date.weekday >= 5:  # Saturday/Sunday
                weekend_discount = 0.95
            else:
                weekend_discount = 1.0

            seasonal_prices[date] = base_price * seasonal_multiplier * weekend_discount

        return {
            'seasonal_prices': seasonal_prices,
            'monthly_averages': seasonal_prices.groupby(seasonal_prices.index.month).mean(),
            'weekly_patterns': seasonal_prices.groupby(seasonal_prices.index.weekday).mean()
        }

    def calculate_storage_value(
        self,
        current_inventory: float,
        max_capacity: float,
        forward_curve: pd.Series,
        storage_costs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate the intrinsic value of storage capacity.

        Args:
            current_inventory: Current storage level (Bcf)
            max_capacity: Maximum storage capacity (Bcf)
            forward_curve: Forward price curve (index by month)
            storage_costs: Override default storage costs

        Returns:
            Dictionary with storage value components
        """
        if storage_costs:
            costs = {**self.storage_costs, **storage_costs}
        else:
            costs = self.storage_costs

        # Available capacity for optimization
        available_capacity = max_capacity - current_inventory

        # Calculate optimal injection/withdrawal strategy
        optimal_strategy = self._optimize_storage_strategy(
            forward_curve, available_capacity, costs
        )

        # Calculate storage value
        storage_value = optimal_strategy['total_profit'] - (
            available_capacity * costs['storage_cost'] * 12  # Annual storage cost
        )

        return {
            'intrinsic_value': storage_value,
            'optimal_strategy': optimal_strategy,
            'available_capacity': available_capacity,
            'utilization_rate': current_inventory / max_capacity,
            'costs': costs
        }

    def _optimize_storage_strategy(
        self,
        forward_curve: pd.Series,
        available_capacity: float,
        costs: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize injection and withdrawal timing for maximum profit."""

        # Simple optimization: inject in summer, withdraw in winter
        summer_months = [6, 7, 8]  # June-August
        winter_months = [12, 1, 2]  # December-February

        # Find summer and winter prices
        summer_prices = forward_curve[forward_curve.index.isin(summer_months)]
        winter_prices = forward_curve[forward_curve.index.isin(winter_months)]

        if len(summer_prices) == 0 or len(winter_prices) == 0:
            return {
                'total_profit': 0,
                'injection_volume': 0,
                'withdrawal_volume': 0,
                'strategy': 'no_opportunity'
            }

        summer_price = summer_prices.mean()
        winter_price = winter_prices.mean()

        # Calculate potential profit per cycle
        injection_volume = min(available_capacity, available_capacity)  # Full capacity
        gross_profit = (winter_price - summer_price) * injection_volume

        # Subtract costs
        injection_cost = injection_volume * costs['injection_cost']
        withdrawal_cost = injection_volume * costs['withdrawal_cost']
        storage_cost = injection_volume * costs['storage_cost'] * 6  # 6 months storage
        fuel_cost = injection_volume * costs['fuel_cost']

        total_costs = injection_cost + withdrawal_cost + storage_cost + fuel_cost
        net_profit = gross_profit - total_costs

        return {
            'total_profit': net_profit,
            'injection_volume': injection_volume,
            'withdrawal_volume': injection_volume,
            'summer_price': summer_price,
            'winter_price': winter_price,
            'gross_profit': gross_profit,
            'costs': total_costs,
            'strategy': 'seasonal_arbitrage' if net_profit > 0 else 'no_opportunity'
        }

    def calculate_time_spread_value(
        self,
        near_month_price: float,
        far_month_price: float,
        storage_period: int,  # months
        costs: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate the value of time spreads for storage arbitrage.

        Args:
            near_month_price: Near month contract price
            far_month_price: Far month contract price
            storage_period: Months between contracts
            costs: Storage costs

        Returns:
            Dictionary with spread value components
        """
        if costs is None:
            costs = self.storage_costs

        # Time spread (contango = positive, backwardation = negative)
        spread = far_month_price - near_month_price
        spread_annualized = spread * (12 / storage_period)

        # Storage costs for the period
        monthly_storage_cost = costs['storage_cost'] * storage_period
        injection_withdrawal_cost = costs['injection_cost'] + costs['withdrawal_cost']

        # Net spread value after costs
        net_spread_value = spread - monthly_storage_cost - injection_withdrawal_cost

        return {
            'spread': spread,
            'spread_annualized': spread_annualized,
            'storage_costs': monthly_storage_cost,
            'injection_withdrawal_costs': injection_withdrawal_cost,
            'net_value': net_spread_value,
            'profitable': net_spread_value > 0
        }

    def optimize_regional_arbitrage(
        self,
        regional_prices: Dict[str, float],
        transport_costs: Dict[str, Dict[str, float]],
        pipeline_capacities: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Optimize arbitrage between different regional gas markets.

        Args:
            regional_prices: Current spot prices by region
            transport_costs: Transport costs between regions
            pipeline_capacities: Available pipeline capacity

        Returns:
            Optimal arbitrage strategy
        """
        regions = list(regional_prices.keys())

        if len(regions) < 2:
            return {'strategy': 'no_opportunity', 'profit': 0}

        # Find price differences
        price_diffs = {}
        for i, region1 in enumerate(regions[:-1]):
            for region2 in regions[i+1:]:
                price_diff = regional_prices[region2] - regional_prices[region1]
                transport_cost = transport_costs.get(region1, {}).get(region2, 0)
                net_diff = price_diff - transport_cost

                if net_diff > 0:  # Profitable arbitrage opportunity
                    capacity = pipeline_capacities.get(region1, {}).get(region2, 0)
                    price_diffs[f'{region1}_to_{region2}'] = {
                        'price_diff': price_diff,
                        'transport_cost': transport_cost,
                        'net_profit': net_diff,
                        'capacity': capacity,
                        'potential_volume': capacity
                    }

        if not price_diffs:
            return {'strategy': 'no_opportunity', 'profit': 0}

        # Select most profitable opportunity
        best_opportunity = max(price_diffs.items(), key=lambda x: x[1]['net_profit'])

        return {
            'strategy': 'regional_arbitrage',
            'from_region': best_opportunity[0].split('_to_')[0],
            'to_region': best_opportunity[0].split('_to_')[1],
            'profit_per_mmbtu': best_opportunity[1]['net_profit'],
            'total_profit': best_opportunity[1]['net_profit'] * best_opportunity[1]['potential_volume'],
            'transport_cost': best_opportunity[1]['transport_cost'],
            'capacity_utilization': best_opportunity[1]['capacity']
        }

    def calculate_storage_capacity_value(
        self,
        forward_curve: pd.Series,
        volatility: float,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate the option value of storage capacity using real options theory.

        Args:
            forward_curve: Forward price curve
            volatility: Price volatility
            risk_free_rate: Risk-free interest rate

        Returns:
            Storage capacity valuation
        """
        # Simplified real options valuation for storage
        # In production: Use more sophisticated models

        # Extract summer and winter prices
        summer_prices = forward_curve[[6, 7, 8]]  # June-August
        winter_prices = forward_curve[[12, 1, 2]]  # December-February

        summer_avg = summer_prices.mean()
        winter_avg = winter_prices.mean()

        # Storage value as option to buy cheap and sell expensive
        price_diff = winter_avg - summer_avg

        # Option value using Black-Scholes approximation
        # Simplified calculation for storage option
        time_to_expiry = 1.0  # 1 year
        d1 = (np.log(winter_avg / summer_avg) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Standard normal CDF approximation
        option_value = (winter_avg * self._normal_cdf(d1) - summer_avg * np.exp(-risk_free_rate * time_to_expiry) * self._normal_cdf(d2))

        return {
            'option_value': option_value,
            'price_differential': price_diff,
            'summer_price': summer_avg,
            'winter_price': winter_avg,
            'volatility': volatility,
            'time_to_expiry': time_to_expiry
        }

    def _normal_cdf(self, x: float) -> float:
        """Approximation of standard normal cumulative distribution function."""
        # Abramowitz and Stegun approximation
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911

        sign = 1 if x > 0 else -1
        x = abs(x) / np.sqrt(2.0)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    def forecast_optimal_inventory(
        self,
        current_inventory: float,
        max_capacity: float,
        price_forecasts: pd.Series,
        injection_capacity: float,
        withdrawal_capacity: float
    ) -> Dict[str, Any]:
        """
        Forecast optimal inventory levels over time.

        Args:
            current_inventory: Current storage level
            max_capacity: Maximum capacity
            price_forecasts: Price forecasts over time
            injection_capacity: Daily injection capacity (MMBtu/day)
            withdrawal_capacity: Daily withdrawal capacity (MMBtu/day)

        Returns:
            Optimal inventory trajectory
        """
        # Simple optimization for inventory management
        # In production: Use dynamic programming or reinforcement learning

        optimal_trajectory = []
        current_inv = current_inventory

        for i, price in enumerate(price_forecasts):
            # Simple rule: inject if price is low, withdraw if price is high
            if price < price_forecasts.mean() * 0.9:  # 10% below average
                # Inject as much as possible
                injection = min(injection_capacity, max_capacity - current_inv)
                current_inv += injection
                action = f'inject_{injection:.1f}'

            elif price > price_forecasts.mean() * 1.1:  # 10% above average
                # Withdraw as much as possible
                withdrawal = min(withdrawal_capacity, current_inv)
                current_inv -= withdrawal
                action = f'withdraw_{withdrawal:.1f}'

            else:
                action = 'hold'

            optimal_trajectory.append({
                'period': i,
                'inventory': current_inv,
                'price': price,
                'action': action,
                'utilization': current_inv / max_capacity
            })

        return {
            'optimal_trajectory': optimal_trajectory,
            'final_inventory': current_inv,
            'average_utilization': np.mean([t['utilization'] for t in optimal_trajectory]),
            'total_injections': sum(t['action'].startswith('inject') for t in optimal_trajectory),
            'total_withdrawals': sum(t['action'].startswith('withdraw') for t in optimal_trajectory)
        }
