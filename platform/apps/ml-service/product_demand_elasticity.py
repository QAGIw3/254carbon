"""
Product Demand Elasticity Model

Model for analyzing price elasticity of demand for refined petroleum products:
- Gasoline demand elasticity
- Diesel demand elasticity
- Jet fuel demand elasticity
- Cross-price elasticities
- Income elasticity effects
- Seasonal demand variations
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ProductDemandElasticity:
    """
    Product demand elasticity analysis model.

    Features:
    - Price elasticity estimation
    - Cross-price elasticity analysis
    - Income elasticity modeling
    - Seasonal demand patterns
    """

    def __init__(self):
        # Base elasticity assumptions (simplified)
        self.base_elasticities = {
            'gasoline': -0.25,     # -0.25 (inelastic)
            'diesel': -0.15,       # -0.15 (more inelastic than gasoline)
            'jet_fuel': -0.10,     # -0.10 (least elastic)
            'heating_oil': -0.20   # -0.20 (similar to gasoline)
        }

        # Cross-price elasticities (how demand for one product responds to price changes in others)
        self.cross_elasticities = {
            'gasoline_diesel': 0.05,    # Gasoline and diesel are weak substitutes
            'gasoline_jet': 0.02,       # Very weak substitution
            'diesel_jet': 0.03         # Weak substitution
        }

        # Income elasticity (demand response to income changes)
        self.income_elasticities = {
            'gasoline': 0.8,     # Positive but less than 1 (necessity)
            'diesel': 0.6,       # Lower income elasticity
            'jet_fuel': 1.2,     # Higher income elasticity (luxury)
            'heating_oil': 0.7   # Moderate income elasticity
        }

    def estimate_price_elasticity(
        self,
        demand_data: pd.Series,
        price_data: pd.Series,
        product_type: str,
        method: str = 'regression'
    ) -> Dict[str, float]:
        """
        Estimate price elasticity of demand for a product.

        Args:
            demand_data: Historical demand data
            price_data: Historical price data
            product_type: Type of product
            method: Estimation method ('regression', 'correlation', 'arc')

        Returns:
            Elasticity estimation results
        """
        # Align data
        data = pd.DataFrame({
            'demand': demand_data,
            'price': price_data
        }).dropna()

        if len(data) < 20:
            return {'error': 'Insufficient data for elasticity estimation'}

        if method == 'regression':
            # Linear regression approach
            X = data[['price']]
            y = data['demand']

            model = LinearRegression()
            model.fit(X, y)

            # Elasticity = coefficient * (mean price / mean demand)
            elasticity = model.coef_[0] * (data['price'].mean() / data['demand'].mean())

            # Calculate R-squared
            r_squared = model.score(X, y)

            return {
                'elasticity': elasticity,
                'coefficient': model.coef_[0],
                'intercept': model.intercept_,
                'r_squared': r_squared,
                'method': 'regression',
                'data_points': len(data)
            }

        elif method == 'correlation':
            # Correlation-based approximation
            correlation = np.corrcoef(data['demand'], data['price'])[0, 1]

            # Convert to elasticity approximation
            demand_std = data['demand'].std()
            price_std = data['price'].std()
            price_mean = data['price'].mean()

            elasticity = correlation * (demand_std / price_std) * (price_mean / data['demand'].mean())

            return {
                'elasticity': elasticity,
                'correlation': correlation,
                'method': 'correlation',
                'data_points': len(data)
            }

        elif method == 'arc':
            # Arc elasticity (midpoint method)
            # Use first and last points for calculation
            if len(data) >= 2:
                first_demand = data['demand'].iloc[0]
                last_demand = data['demand'].iloc[-1]
                first_price = data['price'].iloc[0]
                last_price = data['price'].iloc[-1]

                # Arc elasticity formula
                demand_change_pct = (last_demand - first_demand) / ((first_demand + last_demand) / 2)
                price_change_pct = (last_price - first_price) / ((first_price + last_price) / 2)

                elasticity = demand_change_pct / price_change_pct if price_change_pct != 0 else 0

                return {
                    'elasticity': elasticity,
                    'method': 'arc',
                    'first_demand': first_demand,
                    'last_demand': last_demand,
                    'first_price': first_price,
                    'last_price': last_price,
                    'data_points': len(data)
                }

        else:
            raise ValueError(f"Unknown elasticity method: {method}")

    def estimate_cross_price_elasticity(
        self,
        demand_data: Dict[str, pd.Series],
        price_data: Dict[str, pd.Series],
        product1: str,
        product2: str
    ) -> Dict[str, float]:
        """
        Estimate cross-price elasticity between two products.

        Args:
            demand_data: Demand data for both products
            price_data: Price data for both products
            product1: First product
            product2: Second product

        Returns:
            Cross-price elasticity estimation
        """
        # Align data for both products
        combined_data = pd.DataFrame({
            f'{product1}_demand': demand_data[product1],
            f'{product2}_demand': demand_data[product2],
            f'{product1}_price': price_data[product1],
            f'{product2}_price': price_data[product2]
        }).dropna()

        if len(combined_data) < 20:
            return {'error': 'Insufficient data for cross-elasticity estimation'}

        # Regression: demand1 ~ price1 + price2
        X = combined_data[[f'{product1}_price', f'{product2}_price']]
        y = combined_data[f'{product1}_demand']

        model = LinearRegression()
        model.fit(X, y)

        # Cross-price elasticity is coefficient of other product's price
        own_elasticity = model.coef_[0] * (combined_data[f'{product1}_price'].mean() / combined_data[f'{product1}_demand'].mean())
        cross_elasticity = model.coef_[1] * (combined_data[f'{product2}_price'].mean() / combined_data[f'{product1}_demand'].mean())

        return {
            'own_price_elasticity': own_elasticity,
            'cross_price_elasticity': cross_elasticity,
            'intercept': model.intercept_,
            'r_squared': model.score(X, y),
            'method': 'regression',
            'products': [product1, product2],
            'data_points': len(combined_data)
        }

    def estimate_income_elasticity(
        self,
        demand_data: pd.Series,
        income_data: pd.Series,
        product_type: str
    ) -> Dict[str, float]:
        """
        Estimate income elasticity of demand.

        Args:
            demand_data: Demand data
            income_data: Income or GDP proxy data
            product_type: Type of product

        Returns:
            Income elasticity estimation
        """
        # Align data
        data = pd.DataFrame({
            'demand': demand_data,
            'income': income_data
        }).dropna()

        if len(data) < 20:
            return {'error': 'Insufficient data for income elasticity estimation'}

        # Regression: demand ~ income
        X = data[['income']]
        y = data['demand']

        model = LinearRegression()
        model.fit(X, y)

        # Income elasticity = coefficient * (mean income / mean demand)
        elasticity = model.coef_[0] * (data['income'].mean() / data['demand'].mean())

        return {
            'income_elasticity': elasticity,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'r_squared': model.score(X, y),
            'method': 'regression',
            'product_type': product_type,
            'data_points': len(data)
        }

    def forecast_demand_response(
        self,
        current_demand: Dict[str, float],
        price_changes: Dict[str, float],
        elasticities: Optional[Dict[str, float]] = None,
        cross_elasticities: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Forecast demand response to price changes.

        Args:
            current_demand: Current demand levels by product
            price_changes: Price changes by product (% change)
            elasticities: Own-price elasticities
            cross_elasticities: Cross-price elasticities

        Returns:
            Forecasted demand changes
        """
        if elasticities is None:
            elasticities = self.base_elasticities

        if cross_elasticities is None:
            cross_elasticities = self.cross_elasticities

        forecasted_demand = current_demand.copy()

        for product, current_demand_val in current_demand.items():
            if product not in elasticities:
                continue

            elasticity = elasticities[product]
            price_change = price_changes.get(product, 0)

            # Own-price effect
            demand_change_pct = elasticity * price_change
            forecasted_demand[product] = current_demand_val * (1 + demand_change_pct)

            # Cross-price effects (if applicable)
            for other_product, cross_elasticity in cross_elasticities.items():
                if product in other_product:
                    # Find the other product in the pair
                    if product == other_product.split('_')[0]:
                        other_prod = other_product.split('_')[1]
                    else:
                        other_prod = other_product.split('_')[0]

                    if other_prod in current_demand:
                        other_price_change = price_changes.get(other_prod, 0)
                        cross_demand_change = cross_elasticity * other_price_change
                        forecasted_demand[product] *= (1 + cross_demand_change)

        return forecasted_demand

    def analyze_seasonal_demand_patterns(
        self,
        historical_demand: Dict[str, pd.Series],
        years_back: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Analyze seasonal demand patterns by product.

        Args:
            historical_demand: Historical demand data by product
            years_back: Years of historical data

        Returns:
            Seasonal demand patterns
        """
        seasonal_patterns = {}

        for product, demand_series in historical_demand.items():
            # Filter to analysis period
            end_date = demand_series.index.max()
            start_date = end_date - pd.DateOffset(years=years_back)

            analysis_demand = demand_series[
                (demand_series.index >= start_date) &
                (demand_series.index <= end_date)
            ]

            if len(analysis_demand) == 0:
                continue

            # Monthly demand patterns
            monthly_demand = analysis_demand.groupby(analysis_demand.index.month).agg(['mean', 'std'])

            # Seasonal indices (relative to annual average)
            annual_avg = analysis_demand.mean()
            seasonal_indices = monthly_demand['mean'] / annual_avg

            seasonal_patterns[product] = seasonal_indices

        return seasonal_patterns

    def calculate_demand_sensitivity_matrix(
        self,
        products: List[str],
        price_scenarios: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """
        Calculate demand sensitivity matrix for multiple products and price scenarios.

        Args:
            products: List of products to analyze
            price_scenarios: Price scenarios for each product

        Returns:
            Demand sensitivity matrix
        """
        # Create sensitivity matrix
        sensitivity_matrix = pd.DataFrame(index=products, columns=products)

        for product_i in products:
            for product_j in products:
                if product_i == product_j:
                    # Own-price elasticity
                    sensitivity_matrix.loc[product_i, product_j] = self.base_elasticities.get(product_i, -0.2)
                else:
                    # Cross-price elasticity
                    cross_key = f'{product_i}_{product_j}'
                    reverse_key = f'{product_j}_{product_i}'

                    if cross_key in self.cross_elasticities:
                        sensitivity_matrix.loc[product_i, product_j] = self.cross_elasticities[cross_key]
                    elif reverse_key in self.cross_elasticities:
                        sensitivity_matrix.loc[product_i, product_j] = self.cross_elasticities[reverse_key]
                    else:
                        sensitivity_matrix.loc[product_i, product_j] = 0.0  # No cross-elasticity

        return sensitivity_matrix

    def optimize_pricing_strategy(
        self,
        current_demand: Dict[str, float],
        current_prices: Dict[str, float],
        cost_structure: Dict[str, float],
        elasticity_matrix: pd.DataFrame,
        optimization_target: str = 'profit'
    ) -> Dict[str, Any]:
        """
        Optimize pricing strategy based on demand elasticities.

        Args:
            current_demand: Current demand levels
            current_prices: Current prices
            cost_structure: Cost per unit by product
            elasticity_matrix: Demand elasticity matrix
            optimization_target: 'profit', 'revenue', or 'volume'

        Returns:
            Optimal pricing strategy
        """
        # Simple optimization for demonstration
        # In production: Use more sophisticated optimization

        optimal_prices = {}
        expected_results = {}

        for product, current_price in current_prices.items():
            if product not in elasticity_matrix.index:
                continue

            # Get product-specific elasticity row
            product_elasticities = elasticity_matrix.loc[product]

            # Simple price optimization (assuming linear demand)
            # Optimal price = cost / (1 + 1/|elasticity|)
            elasticity = product_elasticities[product]

            if elasticity < 0:  # Negative elasticity (normal case)
                cost = cost_structure.get(product, current_price * 0.8)
                optimal_price = cost / (1 + 1/abs(elasticity))
            else:
                optimal_price = current_price  # No optimization possible

            optimal_prices[product] = optimal_price

            # Calculate expected outcomes
            price_change = (optimal_price - current_price) / current_price
            demand_change = elasticity * price_change
            new_demand = current_demand[product] * (1 + demand_change)

            if optimization_target == 'profit':
                margin = optimal_price - cost
                expected_profit = new_demand * margin
            elif optimization_target == 'revenue':
                expected_profit = new_demand * optimal_price
            else:  # volume
                expected_profit = new_demand

            expected_results[product] = {
                'optimal_price': optimal_price,
                'price_change': price_change,
                'demand_change': demand_change,
                'new_demand': new_demand,
                'expected_outcome': expected_profit,
                'elasticity_used': elasticity
            }

        # Calculate total expected outcome
        total_expected = sum(result['expected_outcome'] for result in expected_results.values())

        return {
            'optimal_prices': optimal_prices,
            'expected_results': expected_results,
            'total_expected_outcome': total_expected,
            'optimization_target': optimization_target,
            'current_total_outcome': sum(
                current_demand[product] * (current_prices[product] - cost_structure.get(product, 0))
                for product in current_demand.keys()
            ),
            'improvement': total_expected - sum(
                current_demand[product] * (current_prices[product] - cost_structure.get(product, 0))
                for product in current_demand.keys()
            )
        }
