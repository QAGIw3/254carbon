"""
Refinery Yield Modeling System

Advanced model for predicting and optimizing refinery product yields:
- Crude oil assay-based yield prediction
- Process unit optimization
- Product quality specifications
- Yield variability analysis
- Feedstock flexibility assessment
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class RefineryYieldModel:
    """
    Refinery product yield prediction and optimization model.

    Features:
    - Crude assay-based yield prediction
    - Process optimization
    - Quality constraint modeling
    - Yield variability analysis
    """

    def __init__(self):
        # Crude oil assay database (simplified)
        self.crude_assays = {
            'wti': {
                'api_gravity': 39.6,
                'sulfur_content': 0.24,
                'naphtha_yield': 0.25,
                'gasoline_yield': 0.45,
                'diesel_yield': 0.20,
                'fuel_oil_yield': 0.10,
                'viscosity': 2.1,
                'pour_point': -30
            },
            'brent': {
                'api_gravity': 38.3,
                'sulfur_content': 0.37,
                'naphtha_yield': 0.22,
                'gasoline_yield': 0.42,
                'diesel_yield': 0.25,
                'fuel_oil_yield': 0.11,
                'viscosity': 2.3,
                'pour_point': -25
            },
            'dubai': {
                'api_gravity': 31.0,
                'sulfur_content': 2.0,
                'naphtha_yield': 0.15,
                'gasoline_yield': 0.30,
                'diesel_yield': 0.35,
                'fuel_oil_yield': 0.20,
                'viscosity': 4.5,
                'pour_point': 15
            },
            'maya': {
                'api_gravity': 22.0,
                'sulfur_content': 3.5,
                'naphtha_yield': 0.08,
                'gasoline_yield': 0.20,
                'diesel_yield': 0.30,
                'fuel_oil_yield': 0.42,
                'viscosity': 8.2,
                'pour_point': 35
            }
        }

        # Process unit capabilities
        self.process_units = {
            'atmospheric_distillation': {
                'capacity': 200000,  # bbl/day
                'naphtha_yield_range': (0.15, 0.30),
                'gasoline_yield_range': (0.35, 0.55),
                'diesel_yield_range': (0.15, 0.30),
                'fuel_oil_yield_range': (0.05, 0.15)
            },
            'vacuum_distillation': {
                'capacity': 100000,
                'gasoline_yield_range': (0.10, 0.20),
                'diesel_yield_range': (0.20, 0.35),
                'fuel_oil_yield_range': (0.45, 0.70)
            },
            'catalytic_cracking': {
                'capacity': 80000,
                'gasoline_yield_range': (0.40, 0.50),
                'diesel_yield_range': (0.30, 0.40),
                'lpg_yield_range': (0.15, 0.25)
            },
            'hydrocracking': {
                'capacity': 40000,
                'diesel_yield_range': (0.65, 0.75),
                'jet_yield_range': (0.20, 0.30),
                'naphtha_yield_range': (0.05, 0.15)
            }
        }

    def predict_yields_from_assay(
        self,
        crude_type: str,
        process_configuration: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict product yields based on crude oil assay.

        Args:
            crude_type: Type of crude oil
            process_configuration: Process unit utilization rates

        Returns:
            Predicted product yields
        """
        if crude_type not in self.crude_assays:
            raise ValueError(f"Crude assay not available for {crude_type}")

        assay = self.crude_assays[crude_type]

        # Base yields from crude assay
        base_yields = {
            'naphtha': assay['naphtha_yield'],
            'gasoline': assay['gasoline_yield'],
            'diesel': assay['diesel_yield'],
            'fuel_oil': assay['fuel_oil_yield']
        }

        # Apply process unit modifications
        adjusted_yields = base_yields.copy()

        # Atmospheric distillation adjustments
        atm_rate = process_configuration.get('atmospheric_distillation', 1.0)
        if atm_rate > 0:
            # Higher utilization may increase light product yields
            adjusted_yields['naphtha'] *= (1 + 0.02 * (atm_rate - 1))
            adjusted_yields['gasoline'] *= (1 + 0.01 * (atm_rate - 1))

        # Vacuum distillation adjustments
        vac_rate = process_configuration.get('vacuum_distillation', 1.0)
        if vac_rate > 0:
            # Vacuum distillation increases heavy product conversion
            adjusted_yields['fuel_oil'] *= (1 - 0.05 * vac_rate)

        # Catalytic cracking adjustments
        fcc_rate = process_configuration.get('catalytic_cracking', 1.0)
        if fcc_rate > 0:
            # FCC converts heavy products to lighter ones
            conversion_factor = 0.03 * fcc_rate
            adjusted_yields['gasoline'] += conversion_factor * 0.45
            adjusted_yields['diesel'] += conversion_factor * 0.35
            adjusted_yields['fuel_oil'] -= conversion_factor

        # Hydrocracking adjustments
        hc_rate = process_configuration.get('hydrocracking', 1.0)
        if hc_rate > 0:
            # Hydrocracking produces high-quality diesel
            adjusted_yields['diesel'] += 0.02 * hc_rate

        # Normalize yields to sum to 1.0
        total_yield = sum(adjusted_yields.values())
        if total_yield > 0:
            adjusted_yields = {k: v / total_yield for k, v in adjusted_yields.items()}

        return adjusted_yields

    def optimize_process_configuration(
        self,
        crude_type: str,
        product_prices: Dict[str, float],
        operating_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize process unit configuration for maximum value.

        Args:
            crude_type: Type of crude oil being processed
            product_prices: Current product prices
            operating_constraints: Process unit constraints

        Returns:
            Optimal process configuration
        """
        if operating_constraints is None:
            operating_constraints = {
                'max_atmospheric_rate': 1.1,  # 110% of nameplate
                'max_vacuum_rate': 1.05,     # 105% of nameplate
                'max_fcc_rate': 1.15,        # 115% of nameplate
                'max_hydrocracker_rate': 1.0 # 100% of nameplate
            }

        # Get base yields for this crude
        assay = self.crude_assays[crude_type]

        # Simple optimization: maximize high-value products
        # In production: Use linear programming

        # Calculate product values
        product_values = {}
        for product in ['naphtha', 'gasoline', 'diesel', 'fuel_oil']:
            base_yield = assay[f'{product}_yield']
            price = product_prices.get(product, 0)
            value = base_yield * price
            product_values[product] = value

        # Prioritize high-value products
        sorted_products = sorted(product_values.items(), key=lambda x: x[1], reverse=True)

        # Optimal configuration (simplified heuristic)
        optimal_config = {
            'atmospheric_distillation': 1.05,  # Slightly above nameplate
            'vacuum_distillation': 1.0,
            'catalytic_cracking': 1.1,  # Maximize gasoline production
            'hydrocracking': 0.9       # Moderate diesel production
        }

        # Calculate expected yields and value
        expected_yields = self.predict_yields_from_assay(crude_type, optimal_config)

        total_value = sum(
            expected_yields[product] * product_prices.get(product, 0)
            for product in expected_yields.keys()
        )

        # Calculate operating costs
        operating_cost = sum(
            rate * unit['operating_cost'] * unit['capacity'] / 1000
            for unit_name, rate in optimal_config.items()
            for unit in [self.process_units[unit_name]]
        )

        net_value = total_value - operating_cost

        return {
            'optimal_configuration': optimal_config,
            'expected_yields': expected_yields,
            'total_product_value': total_value,
            'operating_cost': operating_cost,
            'net_value': net_value,
            'crude_type': crude_type,
            'value_per_bbl': net_value / 200000,  # Per barrel of throughput
            'constraints_applied': operating_constraints
        }

    def analyze_yield_variability(
        self,
        historical_yields: pd.DataFrame,
        crude_type: str,
        process_unit: str = 'atmospheric_distillation'
    ) -> Dict[str, Any]:
        """
        Analyze variability in product yields.

        Args:
            historical_yields: Historical yield data
            crude_type: Type of crude oil
            process_unit: Process unit to analyze

        Returns:
            Yield variability analysis
        """
        if process_unit not in self.process_units:
            raise ValueError(f"Process unit {process_unit} not found")

        # Filter data for specific process unit and crude type
        unit_yields = historical_yields[
            (historical_yields['process_unit'] == process_unit) &
            (historical_yields['crude_type'] == crude_type)
        ]

        if len(unit_yields) < 30:
            return {'error': 'Insufficient data for variability analysis'}

        # Calculate yield statistics
        yield_stats = {}
        for product in ['naphtha', 'gasoline', 'diesel', 'fuel_oil']:
            if product in unit_yields.columns:
                product_yields = unit_yields[product]
                yield_stats[product] = {
                    'mean': product_yields.mean(),
                    'std': product_yields.std(),
                    'cv': product_yields.std() / product_yields.mean() if product_yields.mean() > 0 else 0,
                    'min': product_yields.min(),
                    'max': product_yields.max(),
                    'range': product_yields.max() - product_yields.min()
                }

        # Correlation analysis between products
        product_correlations = unit_yields[['naphtha', 'gasoline', 'diesel', 'fuel_oil']].corr()

        # Identify periods of high variability
        overall_variability = unit_yields[['naphtha', 'gasoline', 'diesel', 'fuel_oil']].std(axis=1)
        high_variability_periods = overall_variability > overall_variability.quantile(0.75)

        return {
            'yield_statistics': yield_stats,
            'product_correlations': product_correlations,
            'high_variability_periods': high_variability_periods.sum(),
            'average_variability': overall_variability.mean(),
            'process_unit': process_unit,
            'crude_type': crude_type,
            'data_points': len(unit_yields)
        }

    def forecast_yield_trends(
        self,
        historical_yields: pd.DataFrame,
        forecast_horizon: int = 90,
        trend_model: str = 'linear'
    ) -> Dict[str, pd.Series]:
        """
        Forecast future yield trends.

        Args:
            historical_yields: Historical yield data
            forecast_horizon: Days to forecast
            trend_model: Trend model type ('linear', 'exponential', 'seasonal')

        Returns:
            Yield trend forecasts by product
        """
        forecasts = {}

        products = ['naphtha', 'gasoline', 'diesel', 'fuel_oil']

        for product in products:
            if product not in historical_yields.columns:
                continue

            product_yields = historical_yields[product].dropna()

            if len(product_yields) < 30:
                # Insufficient data - use mean
                forecast_values = [product_yields.mean()] * forecast_horizon
            else:
                if trend_model == 'linear':
                    # Simple linear trend
                    x = np.arange(len(product_yields))
                    y = product_yields.values

                    # Fit linear regression
                    slope, intercept = np.polyfit(x, y, 1)

                    # Forecast
                    forecast_x = np.arange(len(product_yields), len(product_yields) + forecast_horizon)
                    forecast_values = slope * forecast_x + intercept

                elif trend_model == 'exponential':
                    # Exponential trend (for growth patterns)
                    x = np.arange(len(product_yields))
                    y = product_yields.values

                    # Fit exponential curve
                    try:
                        # Log-transform for linear fit
                        log_y = np.log(y + 1e-10)  # Add small constant
                        slope, intercept = np.polyfit(x, log_y, 1)

                        # Forecast
                        forecast_x = np.arange(len(product_yields), len(product_yields) + forecast_horizon)
                        log_forecast = slope * forecast_x + intercept
                        forecast_values = np.exp(log_forecast) - 1e-10
                    except:
                        # Fallback to linear if exponential fails
                        forecast_values = [product_yields.mean()] * forecast_horizon

                else:
                    # Default to mean
                    forecast_values = [product_yields.mean()] * forecast_horizon

            # Create forecast series
            last_date = historical_yields.index[-1]
            forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
            forecasts[product] = pd.Series(forecast_values, index=forecast_dates)

        return forecasts

    def calculate_quality_adjustments(
        self,
        product_yields: Dict[str, float],
        quality_specifications: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate quality-based yield adjustments.

        Args:
            product_yields: Base product yields
            quality_specifications: Quality requirements by product

        Returns:
            Quality-adjusted yields
        """
        adjusted_yields = product_yields.copy()

        for product, specs in quality_specifications.items():
            if product not in adjusted_yields:
                continue

            base_yield = adjusted_yields[product]

            # Quality adjustments (simplified)
            quality_penalty = 0

            # Sulfur content adjustment
            max_sulfur = specs.get('max_sulfur', 0.001)  # ppm
            if max_sulfur < 10:  # Ultra-low sulfur requirement
                quality_penalty += 0.02  # 2% yield penalty

            # Octane requirement adjustment
            min_octane = specs.get('min_octane', 87)
            if min_octane > 90:  # Premium requirement
                quality_penalty += 0.01  # 1% yield penalty

            # Apply quality penalty
            adjusted_yields[product] = base_yield * (1 - quality_penalty)

        # Re-normalize after adjustments
        total_adjusted = sum(adjusted_yields.values())
        if total_adjusted > 0:
            adjusted_yields = {k: v / total_adjusted for k, v in adjusted_yields.items()}

        return adjusted_yields

    def assess_feedstock_flexibility(
        self,
        crude_inventory: Dict[str, float],
        product_demand: Dict[str, float],
        quality_requirements: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Assess refinery flexibility with different feedstocks.

        Args:
            crude_inventory: Available crude by type
            product_demand: Required product outputs
            quality_requirements: Quality specifications

        Returns:
            Feedstock flexibility assessment
        """
        flexibility_scores = {}

        for crude_type, inventory in crude_inventory.items():
            if inventory <= 0:
                continue

            # Predict yields for this crude
            yields = self.predict_yields_from_assay(crude_type, {
                'atmospheric_distillation': 1.0,
                'vacuum_distillation': 1.0,
                'catalytic_cracking': 1.0,
                'hydrocracking': 1.0
            })

            # Apply quality adjustments
            quality_adjusted_yields = self.calculate_quality_adjustments(yields, quality_requirements)

            # Calculate how well this crude meets demand
            demand_satisfaction = {}
            for product, required in product_demand.items():
                if product in quality_adjusted_yields:
                    available = quality_adjusted_yields[product] * inventory
                    satisfaction_ratio = min(1.0, available / required) if required > 0 else 1.0
                    demand_satisfaction[product] = satisfaction_ratio

            # Overall flexibility score
            avg_satisfaction = np.mean(list(demand_satisfaction.values()))
            min_satisfaction = min(demand_satisfaction.values())

            flexibility_scores[crude_type] = {
                'predicted_yields': yields,
                'quality_adjusted_yields': quality_adjusted_yields,
                'demand_satisfaction': demand_satisfaction,
                'flexibility_score': avg_satisfaction,
                'min_satisfaction': min_satisfaction,
                'inventory_available': inventory,
                'meets_demand': min_satisfaction >= 0.8  # 80% satisfaction threshold
            }

        # Find most flexible crude
        best_crude = max(flexibility_scores.items(), key=lambda x: x[1]['flexibility_score'])

        return {
            'flexibility_analysis': flexibility_scores,
            'most_flexible_crude': best_crude[0],
            'best_flexibility_score': best_crude[1]['flexibility_score'],
            'total_inventory': sum(crude_inventory.values()),
            'quality_requirements': quality_requirements,
            'product_demand': product_demand
        }
