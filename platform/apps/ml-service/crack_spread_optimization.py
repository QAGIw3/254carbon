"""
Crack Spread Optimization Model

Advanced model for optimizing refinery crack spreads and product yields:
- 3:2:1 crack spread optimization
- Refinery yield optimization
- Product slate optimization
- Margin maximization strategies
- Feedstock selection and blending
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CrackSpreadOptimizer:
    """
    Refinery crack spread optimization model.

    Features:
    - Multi-crack spread optimization (3:2:1, 5:3:2, etc.)
    - Product yield optimization
    - Feedstock economics analysis
    - Margin maximization strategies
    """

    def __init__(self):
        # Refinery configuration parameters
        self.refinery_config = {
            'atmospheric_distillation': {
                'capacity': 200000,  # bbl/day
                'efficiency': 0.98,
                'operating_cost': 1.50  # $/bbl
            },
            'vacuum_distillation': {
                'capacity': 100000,
                'efficiency': 0.96,
                'operating_cost': 2.00
            },
            'catalytic_cracking': {
                'capacity': 80000,
                'gasoline_yield': 0.45,  # 45% gasoline
                'diesel_yield': 0.35,    # 35% diesel
                'operating_cost': 3.50
            },
            'hydrocracking': {
                'capacity': 40000,
                'diesel_yield': 0.70,
                'jet_yield': 0.25,
                'operating_cost': 4.00
            },
            'reforming': {
                'capacity': 60000,
                'gasoline_yield': 0.85,
                'operating_cost': 2.50
            }
        }

        # Product specifications and yields
        self.product_yields = {
            'naphtha': {'yield': 0.05, 'value_multiplier': 1.0},
            'gasoline': {'yield': 0.45, 'value_multiplier': 1.1},
            'jet_fuel': {'yield': 0.15, 'value_multiplier': 1.05},
            'diesel': {'yield': 0.25, 'value_multiplier': 1.0},
            'fuel_oil': {'yield': 0.10, 'value_multiplier': 0.8}
        }

    def calculate_crack_spread(
        self,
        crude_price: float,
        gasoline_price: float,
        diesel_price: float,
        jet_price: float = None,
        crack_type: str = "3:2:1"
    ) -> Dict[str, float]:
        """
        Calculate crack spread value.

        Args:
            crude_price: Crude oil price ($/bbl)
            gasoline_price: Gasoline price ($/gallon)
            diesel_price: Diesel price ($/gallon)
            jet_price: Jet fuel price ($/gallon)
            crack_type: Type of crack spread calculation

        Returns:
            Crack spread analysis
        """
        # Convert gasoline and diesel to $/bbl
        gasoline_bbl = gasoline_price * 42  # 42 gallons per barrel
        diesel_bbl = diesel_price * 42

        if crack_type == "3:2:1":
            # 3 barrels crude -> 2 barrels gasoline + 1 barrel diesel
            product_value = 2 * gasoline_bbl + 1 * diesel_bbl
            crack_spread = product_value - 3 * crude_price

        elif crack_type == "5:3:2":
            # 5 barrels crude -> 3 barrels gasoline + 2 barrels diesel
            product_value = 3 * gasoline_bbl + 2 * diesel_bbl
            crack_spread = product_value - 5 * crude_price

        elif crack_type == "2:1:1":
            # 2 barrels crude -> 1 barrel gasoline + 1 barrel diesel
            product_value = gasoline_bbl + diesel_bbl
            crack_spread = product_value - 2 * crude_price

        elif crack_type == "3:2:1:1" and jet_price:
            # 3:2:1:1 crack with jet fuel
            jet_bbl = jet_price * 42
            product_value = 2 * gasoline_bbl + 1 * diesel_bbl + 1 * jet_bbl
            crack_spread = product_value - 3 * crude_price

        else:
            raise ValueError(f"Unknown crack spread type: {crack_type}")

        # Calculate per-barrel crack spread
        crack_per_bbl = crack_spread / 3  # Normalized per barrel of crude

        return {
            'crack_spread': crack_spread,
            'crack_per_bbl': crack_per_bbl,
            'product_value': product_value,
            'crude_cost': 3 * crude_price,
            'gasoline_bbl': gasoline_bbl,
            'diesel_bbl': diesel_bbl,
            'jet_bbl': jet_bbl if jet_price else 0,
            'crack_type': crack_type,
            'profitable': crack_spread > 0
        }

    def optimize_product_slate(
        self,
        crude_prices: Dict[str, float],
        product_prices: Dict[str, float],
        refinery_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize refinery product slate for maximum margin.

        Args:
            crude_prices: Crude prices by type ($/bbl)
            product_prices: Product prices ($/gallon)
            refinery_constraints: Refinery operating constraints

        Returns:
            Optimal product slate and margins
        """
        # Default constraints if not provided
        if refinery_constraints is None:
            refinery_constraints = {
                'min_gasoline': 0.35,  # Minimum 35% gasoline
                'max_gasoline': 0.55,  # Maximum 55% gasoline
                'min_diesel': 0.20,    # Minimum 20% diesel
                'max_diesel': 0.40,    # Maximum 40% diesel
                'jet_fuel_ratio': 0.15 # Jet fuel as % of diesel
            }

        # Calculate crack spreads for different crude types
        crack_analyses = {}
        for crude_type, crude_price in crude_prices.items():
            crack_analyses[crude_type] = self.calculate_crack_spread(
                crude_price,
                product_prices.get('gasoline', 2.50),
                product_prices.get('diesel', 2.60),
                product_prices.get('jet_fuel', 2.55)
            )

        # Find optimal crude selection (highest crack spread)
        best_crude = max(crack_analyses.items(), key=lambda x: x[1]['crack_spread'])

        # Optimize product yields within constraints
        optimal_yields = self._optimize_yields(
            product_prices,
            refinery_constraints,
            crack_analyses[best_crude[0]]['crack_spread']
        )

        # Calculate total refinery margin
        crude_throughput = 200000  # bbl/day
        total_margin = crack_analyses[best_crude[0]]['crack_spread'] * (crude_throughput / 3)

        return {
            'optimal_crude': best_crude[0],
            'optimal_crude_price': best_crude[1]['crude_cost'] / 3,
            'crack_spread_analysis': crack_analyses,
            'optimal_product_yields': optimal_yields,
            'total_daily_margin': total_margin,
            'margin_per_bbl': total_margin / crude_throughput,
            'refinery_throughput': crude_throughput,
            'constraints_applied': refinery_constraints
        }

    def _optimize_yields(
        self,
        product_prices: Dict[str, float],
        constraints: Dict[str, float],
        base_crack: float
    ) -> Dict[str, float]:
        """Optimize product yields within operational constraints."""

        # Simple optimization: maximize high-value products within constraints
        # In production: Use linear programming for more sophisticated optimization

        # Base yields from typical refinery configuration
        base_yields = {
            'gasoline': 0.45,
            'diesel': 0.25,
            'jet_fuel': 0.15,
            'fuel_oil': 0.10,
            'naphtha': 0.05
        }

        # Calculate product values (price * yield)
        product_values = {}
        for product, yield_pct in base_yields.items():
            price = product_prices.get(product, 0)
            value = price * yield_pct
            product_values[product] = value

        # Identify highest value products
        sorted_products = sorted(product_values.items(), key=lambda x: x[1], reverse=True)

        # Allocate yields within constraints
        optimized_yields = base_yields.copy()

        # Increase high-value products up to max constraints
        for product, _ in sorted_products[:3]:  # Top 3 products
            current_yield = optimized_yields[product]
            max_yield = constraints.get(f'max_{product}', current_yield + 0.10)

            if current_yield < max_yield:
                # Increase yield by 5% (within operating limits)
                increase = min(0.05, max_yield - current_yield)
                optimized_yields[product] += increase

                # Decrease low-value products to compensate
                for low_product in sorted_products[-2:]:  # Bottom 2 products
                    low_product_name = low_product[0]
                    if optimized_yields[low_product_name] > 0.02:  # Minimum 2% yield
                        decrease = min(increase * 0.5, optimized_yields[low_product_name] - 0.02)
                        optimized_yields[low_product_name] -= decrease
                        increase -= decrease
                        if increase <= 0:
                            break

        # Ensure constraints are met
        for product in ['gasoline', 'diesel']:
            min_constraint = constraints.get(f'min_{product}', 0)
            max_constraint = constraints.get(f'max_{product}', 1)

            optimized_yields[product] = max(min_constraint, min(max_constraint, optimized_yields[product]))

        # Normalize to ensure yields sum to 1.0
        total_yield = sum(optimized_yields.values())
        if total_yield > 0:
            optimized_yields = {k: v / total_yield for k, v in optimized_yields.items()}

        return optimized_yields

    def calculate_feedstock_economics(
        self,
        crude_types: List[str],
        crude_prices: Dict[str, float],
        product_prices: Dict[str, float],
        transport_costs: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Analyze economics of different crude feedstocks.

        Args:
            crude_types: List of crude oil types
            crude_prices: Prices by crude type ($/bbl)
            product_prices: Product prices ($/gallon)
            transport_costs: Transport costs by crude type

        Returns:
            Feedstock economics analysis
        """
        if transport_costs is None:
            transport_costs = {crude: 0 for crude in crude_types}

        feedstock_analysis = {}

        for crude_type in crude_types:
            crude_price = crude_prices[crude_type]
            transport_cost = transport_costs.get(crude_type, 0)
            delivered_cost = crude_price + transport_cost

            # Calculate crack spread for this crude
            crack_analysis = self.calculate_crack_spread(
                delivered_cost,
                product_prices.get('gasoline', 2.50),
                product_prices.get('diesel', 2.60)
            )

            # Calculate net margin per barrel
            margin_per_bbl = crack_analysis['crack_per_bbl']

            feedstock_analysis[crude_type] = {
                'crude_price': crude_price,
                'transport_cost': transport_cost,
                'delivered_cost': delivered_cost,
                'crack_spread': crack_analysis['crack_spread'],
                'margin_per_bbl': margin_per_bbl,
                'margin_rank': 0,  # Will be set after analysis
                'economic': margin_per_bbl > 0
            }

        # Rank feedstocks by margin
        ranked_feedstocks = sorted(
            feedstock_analysis.items(),
            key=lambda x: x[1]['margin_per_bbl'],
            reverse=True
        )

        for i, (crude_type, _) in enumerate(ranked_feedstocks):
            feedstock_analysis[crude_type]['margin_rank'] = i + 1

        return {
            'feedstock_analysis': feedstock_analysis,
            'ranked_feedstocks': ranked_feedstocks,
            'best_feedstock': ranked_feedstocks[0][0] if ranked_feedstocks else None,
            'worst_feedstock': ranked_feedstocks[-1][0] if ranked_feedstocks else None,
            'average_margin': np.mean([f['margin_per_bbl'] for f in feedstock_analysis.values()]),
            'margin_range': max([f['margin_per_bbl'] for f in feedstock_analysis.values()]) - min([f['margin_per_bbl'] for f in feedstock_analysis.values()])
        }

    def forecast_crack_spread_volatility(
        self,
        historical_crack_data: pd.Series,
        forecast_horizon: int = 30,
        volatility_model: str = 'garch'
    ) -> pd.Series:
        """
        Forecast crack spread volatility for risk management.

        Args:
            historical_crack_data: Historical crack spread data
            forecast_horizon: Days to forecast
            volatility_model: Volatility model type

        Returns:
            Volatility forecast series
        """
        if volatility_model == 'garch':
            # Simple GARCH(1,1) approximation
            returns = historical_crack_data.pct_change().dropna()

            # Estimate GARCH parameters (simplified)
            omega = returns.var() * 0.1  # Long-run variance
            alpha = 0.15  # ARCH parameter
            beta = 0.80   # GARCH parameter

            # Current variance
            current_var = returns.iloc[-1] ** 2

            # Forecast variance
            forecast_dates = pd.date_range(
                historical_crack_data.index[-1] + pd.Timedelta(days=1),
                periods=forecast_horizon
            )

            variance_forecast = []
            var_t = current_var

            for _ in range(forecast_horizon):
                var_t = omega + alpha * (returns.iloc[-1] ** 2) + beta * var_t
                variance_forecast.append(var_t)

            volatility_forecast = np.sqrt(variance_forecast)

        elif volatility_model == 'rolling_std':
            # Rolling standard deviation
            window = min(30, len(historical_crack_data) // 4)
            volatility = historical_crack_data.rolling(window).std()

            recent_volatility = volatility.iloc[-window:].mean()
            volatility_forecast = pd.Series([recent_volatility] * forecast_horizon,
                                          index=pd.date_range(
                                              historical_crack_data.index[-1] + pd.Timedelta(days=1),
                                              periods=forecast_horizon
                                          ))

        else:
            # Default to recent average volatility
            recent_volatility = historical_crack_data.iloc[-30:].std()
            volatility_forecast = pd.Series([recent_volatility] * forecast_horizon,
                                          index=pd.date_range(
                                              historical_crack_data.index[-1] + pd.Timedelta(days=1),
                                              periods=forecast_horizon
                                          ))

        return pd.Series(volatility_forecast, index=forecast_dates)

    def optimize_refinery_operations(
        self,
        crude_inventory: Dict[str, float],
        product_demand: Dict[str, float],
        operating_costs: Dict[str, float],
        market_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Optimize refinery operations for maximum profitability.

        Args:
            crude_inventory: Available crude inventory by type (bbl)
            product_demand: Required product outputs
            operating_costs: Operating costs by process
            market_prices: Current market prices

        Returns:
            Optimal operating strategy
        """
        # Simple optimization for demonstration
        # In production: Use more sophisticated optimization algorithms

        # Calculate processing costs
        total_processing_cost = sum(operating_costs.values())

        # Calculate potential revenue from products
        total_product_value = sum(
            demand * price for product, demand in product_demand.items()
            for price in [market_prices.get(product, 0)]
        )

        # Calculate available crude value
        total_crude_value = sum(
            inventory * price for crude_type, inventory in crude_inventory.items()
            for price in [market_prices.get(f'crude_{crude_type}', 0)]
        )

        # Net margin calculation
        gross_margin = total_product_value - total_crude_value
        net_margin = gross_margin - total_processing_cost

        # Determine if operations are profitable
        profitable_operations = net_margin > 0

        # Optimize crude utilization
        total_crude_available = sum(crude_inventory.values())
        crude_utilization_rate = min(1.0, total_crude_available / 200000)  # 200k bbl/day capacity

        return {
            'profitable_operations': profitable_operations,
            'gross_margin': gross_margin,
            'net_margin': net_margin,
            'processing_cost': total_processing_cost,
            'product_revenue': total_product_value,
            'crude_cost': total_crude_value,
            'crude_utilization': crude_utilization_rate,
            'total_crude_inventory': total_crude_available,
            'refinery_capacity': 200000,
            'margin_per_bbl': net_margin / total_crude_available if total_crude_available > 0 else 0
        }

    def calculate_seasonal_crack_patterns(
        self,
        historical_crack_data: pd.Series,
        years_back: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Analyze seasonal crack spread patterns.

        Args:
            historical_crack_data: Historical crack spread data
            years_back: Years of historical data to analyze

        Returns:
            Seasonal crack spread patterns
        """
        # Filter to analysis period
        end_date = historical_crack_data.index.max()
        start_date = end_date - pd.DateOffset(years=years_back)

        analysis_data = historical_crack_data[
            (historical_crack_data.index >= start_date) &
            (historical_crack_data.index <= end_date)
        ]

        if len(analysis_data) == 0:
            return {'error': 'No data in analysis period'}

        # Monthly crack spread patterns
        monthly_cracks = analysis_data.groupby(analysis_data.index.month).agg(['mean', 'std', 'min', 'max'])

        # Seasonal volatility patterns
        monthly_volatility = monthly_cracks['std']

        # Identify best and worst months
        best_month = monthly_cracks['mean'].idxmax()
        worst_month = monthly_cracks['mean'].idxmin()

        return {
            'monthly_patterns': monthly_cracks['mean'],
            'monthly_volatility': monthly_volatility,
            'best_month': best_month,
            'worst_month': worst_month,
            'seasonal_range': monthly_cracks['max'].max() - monthly_cracks['min'].min(),
            'analysis_period': f'{start_date.date()} to {end_date.date()}',
            'data_points': len(analysis_data)
        }
