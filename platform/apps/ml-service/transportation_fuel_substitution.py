"""
Transportation Fuel Substitution Model

Model for analyzing substitution between transportation fuels:
- Gasoline vs diesel demand analysis
- Electric vehicle adoption impact
- Biofuel substitution economics
- Cross-fuel price sensitivity
- Infrastructure constraint modeling
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class TransportationFuelSubstitution:
    """
    Transportation fuel substitution analysis model.

    Features:
    - Gasoline-diesel substitution analysis
    - Electric vehicle impact modeling
    - Biofuel adoption forecasting
    - Infrastructure constraint analysis
    """

    def __init__(self):
        # Fuel substitution parameters
        self.substitution_factors = {
            'gasoline_diesel': 0.05,     # 5% substitution rate
            'gasoline_electric': 0.02,   # 2% EV substitution rate
            'diesel_electric': 0.03,     # 3% EV substitution rate
            'gasoline_biofuels': 0.08,   # 8% biofuel substitution rate
            'diesel_biofuels': 0.06      # 6% biofuel substitution rate
        }

        # Fuel characteristics
        self.fuel_characteristics = {
            'gasoline': {
                'energy_density': 32.0,     # MJ/L
                'co2_emissions': 2.3,       # kg CO2/L
                'price_volatility': 0.15,   # 15% annual volatility
                'infrastructure': 'extensive'
            },
            'diesel': {
                'energy_density': 35.8,     # MJ/L
                'co2_emissions': 2.7,       # kg CO2/L
                'price_volatility': 0.12,   # 12% annual volatility
                'infrastructure': 'extensive'
            },
            'electric': {
                'energy_density': 0.0,      # Not applicable
                'co2_emissions': 0.0,       # Depends on generation mix
                'price_volatility': 0.08,   # 8% annual volatility
                'infrastructure': 'developing'
            },
            'ethanol': {
                'energy_density': 21.1,     # MJ/L (lower than gasoline)
                'co2_emissions': 1.4,       # kg CO2/L (lower than gasoline)
                'price_volatility': 0.18,   # 18% annual volatility
                'infrastructure': 'limited'
            },
            'biodiesel': {
                'energy_density': 32.9,     # MJ/L (similar to diesel)
                'co2_emissions': 1.8,       # kg CO2/L (lower than diesel)
                'price_volatility': 0.16,   # 16% annual volatility
                'infrastructure': 'limited'
            }
        }

    def analyze_gasoline_diesel_substitution(
        self,
        gasoline_demand: pd.Series,
        diesel_demand: pd.Series,
        gasoline_prices: pd.Series,
        diesel_prices: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze substitution between gasoline and diesel.

        Args:
            gasoline_demand: Gasoline demand data
            diesel_demand: Diesel demand data
            gasoline_prices: Gasoline price data
            diesel_prices: Diesel price data

        Returns:
            Substitution analysis results
        """
        # Align data
        data = pd.DataFrame({
            'gasoline_demand': gasoline_demand,
            'diesel_demand': diesel_demand,
            'gasoline_price': gasoline_prices,
            'diesel_price': diesel_prices
        }).dropna()

        if len(data) < 30:
            return {'error': 'Insufficient data for substitution analysis'}

        # Calculate relative prices
        data['gasoline_diesel_ratio'] = data['gasoline_price'] / data['diesel_price']

        # Calculate demand correlation
        demand_correlation = np.corrcoef(data['gasoline_demand'], data['diesel_demand'])[0, 1]

        # Regression analysis for substitution
        X = data[['gasoline_diesel_ratio']]
        y = data['gasoline_demand'] / data['diesel_demand']

        model = LinearRegression()
        model.fit(X, y)

        substitution_coefficient = model.coef_[0]

        # Calculate substitution elasticity
        avg_ratio = data['gasoline_diesel_ratio'].mean()
        avg_demand_ratio = (data['gasoline_demand'] / data['diesel_demand']).mean()

        if avg_ratio > 0 and avg_demand_ratio > 0:
            substitution_elasticity = substitution_coefficient * (avg_ratio / avg_demand_ratio)
        else:
            substitution_elasticity = 0

        return {
            'demand_correlation': demand_correlation,
            'substitution_coefficient': substitution_coefficient,
            'substitution_elasticity': substitution_elasticity,
            'average_gasoline_diesel_ratio': avg_ratio,
            'average_demand_ratio': avg_demand_ratio,
            'r_squared': model.score(X, y),
            'data_points': len(data)
        }

    def forecast_electric_vehicle_impact(
        self,
        current_gasoline_demand: float,
        ev_adoption_rate: float,
        ev_efficiency_improvement: float = 0.02,  # 2% annual efficiency improvement
        forecast_years: int = 10
    ) -> Dict[str, Any]:
        """
        Forecast impact of electric vehicle adoption on gasoline demand.

        Args:
            current_gasoline_demand: Current gasoline demand (gallons/year)
            ev_adoption_rate: Annual EV adoption rate (fraction)
            ev_efficiency_improvement: Annual efficiency improvement rate
            forecast_years: Years to forecast

        Returns:
            EV impact forecast
        """
        # Simple exponential decay model for gasoline demand
        years = np.arange(forecast_years + 1)

        # Gasoline demand reduction due to EV adoption
        gasoline_demand_forecast = []
        cumulative_ev_impact = 0

        for year in years:
            # EV adoption follows logistic curve
            ev_penetration = 1 / (1 + np.exp(-ev_adoption_rate * year))

            # Demand reduction = EV penetration * efficiency factor
            demand_reduction = ev_penetration * (1 + ev_efficiency_improvement * year)

            # Apply to current demand
            remaining_demand = current_gasoline_demand * (1 - demand_reduction)
            gasoline_demand_forecast.append(remaining_demand)

            cumulative_ev_impact += demand_reduction

        return {
            'gasoline_demand_forecast': gasoline_demand_forecast,
            'cumulative_impact': cumulative_ev_impact,
            'final_year_demand': gasoline_demand_forecast[-1],
            'demand_reduction_pct': (1 - gasoline_demand_forecast[-1] / current_gasoline_demand) * 100,
            'ev_adoption_rate': ev_adoption_rate,
            'forecast_years': forecast_years
        }

    def analyze_biofuel_substitution_economics(
        self,
        gasoline_demand: pd.Series,
        ethanol_prices: pd.Series,
        gasoline_prices: pd.Series,
        blend_limits: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Analyze economics of ethanol substitution for gasoline.

        Args:
            gasoline_demand: Gasoline demand data
            ethanol_prices: Ethanol price data
            gasoline_prices: Gasoline price data
            blend_limits: Maximum blend percentages by region

        Returns:
            Biofuel substitution analysis
        """
        if blend_limits is None:
            blend_limits = {
                'us': 0.15,      # 15% ethanol blend limit (E15)
                'europe': 0.10,  # 10% ethanol blend limit (E10)
                'brazil': 0.27   # 27% ethanol blend limit (E27)
            }

        # Align data
        data = pd.DataFrame({
            'gasoline_demand': gasoline_demand,
            'ethanol_price': ethanol_prices,
            'gasoline_price': gasoline_prices
        }).dropna()

        if len(data) < 20:
            return {'error': 'Insufficient data for biofuel analysis'}

        # Calculate ethanol-gasoline price ratio
        data['ethanol_gasoline_ratio'] = data['ethanol_price'] / data['gasoline_price']

        # Analyze substitution economics
        # Ethanol is competitive when priced below gasoline equivalent value
        avg_ratio = data['ethanol_gasoline_ratio'].mean()

        # Calculate economic substitution potential
        # Ethanol has lower energy content (about 67% of gasoline)
        energy_equivalent_ratio = 0.67

        economic_threshold = energy_equivalent_ratio * avg_ratio

        # Calculate potential substitution volume
        max_blend_pct = max(blend_limits.values())
        potential_substitution_volume = gasoline_demand.mean() * max_blend_pct

        return {
            'ethanol_gasoline_price_ratio': avg_ratio,
            'energy_equivalent_threshold': energy_equivalent_ratio,
            'economic_substitution_threshold': economic_threshold,
            'blend_limits': blend_limits,
            'potential_substitution_volume': potential_substitution_volume,
            'current_avg_demand': gasoline_demand.mean(),
            'data_points': len(data)
        }

    def model_infrastructure_constraints(
        self,
        current_demand: Dict[str, float],
        infrastructure_capacity: Dict[str, float],
        growth_rates: Dict[str, float],
        constraint_years: int = 5
    ) -> Dict[str, Any]:
        """
        Model infrastructure constraints on fuel substitution.

        Args:
            current_demand: Current demand by fuel type
            infrastructure_capacity: Infrastructure capacity by fuel type
            growth_rates: Annual growth rates by fuel type
            constraint_years: Years to analyze constraints

        Returns:
            Infrastructure constraint analysis
        """
        constraint_analysis = {}

        for fuel_type, current_demand_val in current_demand.items():
            capacity = infrastructure_capacity.get(fuel_type, current_demand_val * 2)

            # Forecast demand
            years = np.arange(constraint_years + 1)
            demand_forecast = current_demand_val * (1 + growth_rates.get(fuel_type, 0.02)) ** years

            # Check capacity constraints
            capacity_exceeded = []
            for year, demand in enumerate(demand_forecast):
                if demand > capacity:
                    capacity_exceeded.append({
                        'year': year,
                        'demand': demand,
                        'capacity': capacity,
                        'shortfall': demand - capacity,
                        'shortfall_pct': (demand - capacity) / capacity * 100
                    })

            constraint_analysis[fuel_type] = {
                'current_demand': current_demand_val,
                'capacity': capacity,
                'capacity_utilization': current_demand_val / capacity,
                'capacity_exceeded_in': len(capacity_exceeded),
                'first_constraint_year': capacity_exceeded[0]['year'] if capacity_exceeded else None,
                'max_shortfall': max([c['shortfall'] for c in capacity_exceeded]) if capacity_exceeded else 0,
                'growth_rate': growth_rates.get(fuel_type, 0.02)
            }

        # Overall system analysis
        total_current_demand = sum(current_demand.values())
        total_capacity = sum(infrastructure_capacity.values())

        return {
            'constraint_analysis': constraint_analysis,
            'total_current_demand': total_current_demand,
            'total_capacity': total_capacity,
            'system_utilization': total_current_demand / total_capacity,
            'constraint_years': constraint_years,
            'fuels_analyzed': list(current_demand.keys())
        }

    def forecast_cross_fuel_substitution(
        self,
        fuel_demand: Dict[str, pd.Series],
        fuel_prices: Dict[str, pd.Series],
        substitution_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        """
        Forecast substitution between multiple fuel types.

        Args:
            fuel_demand: Historical demand data by fuel type
            fuel_prices: Historical price data by fuel type
            substitution_matrix: Substitution elasticities between fuels

        Returns:
            Cross-fuel substitution forecasts
        """
        if substitution_matrix is None:
            # Default substitution matrix
            fuels = list(fuel_demand.keys())
            substitution_matrix = pd.DataFrame(0, index=fuels, columns=fuels)

            # Set diagonal (own-price elasticities)
            for fuel in fuels:
                substitution_matrix.loc[fuel, fuel] = self.base_elasticities.get(fuel, -0.2)

            # Set cross-price elasticities
            for i, fuel1 in enumerate(fuels[:-1]):
                for fuel2 in fuels[i+1:]:
                    key = f'{fuel1}_{fuel2}'
                    if key in self.cross_elasticities:
                        substitution_matrix.loc[fuel1, fuel2] = self.cross_elasticities[key]
                        substitution_matrix.loc[fuel2, fuel1] = self.cross_elasticities[key]

        # Simple forecasting model (could be enhanced with more sophisticated methods)
        forecasts = {}

        for fuel, demand_series in fuel_demand.items():
            # Use recent trend for forecasting
            recent_demand = demand_series.iloc[-90:]  # Last 90 days

            # Simple linear trend
            x = np.arange(len(recent_demand))
            y = recent_demand.values

            if len(x) > 10:
                slope, intercept = np.polyfit(x, y, 1)
                forecast_values = slope * np.arange(len(recent_demand), len(recent_demand) + 30) + intercept
            else:
                forecast_values = [recent_demand.mean()] * 30

            # Create forecast series
            last_date = demand_series.index[-1]
            forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)
            forecasts[fuel] = pd.Series(forecast_values, index=forecast_dates)

        return forecasts

    def calculate_substitution_economics(
        self,
        fuel1_demand: float,
        fuel2_demand: float,
        fuel1_price: float,
        fuel2_price: float,
        substitution_rate: float,
        fuel1_cost: float,
        fuel2_cost: float
    ) -> Dict[str, float]:
        """
        Calculate economics of fuel substitution.

        Args:
            fuel1_demand: Demand for fuel 1
            fuel2_demand: Demand for fuel 2
            fuel1_price: Price of fuel 1
            fuel2_price: Price of fuel 2
            substitution_rate: Rate of substitution
            fuel1_cost: Production cost of fuel 1
            fuel2_cost: Production cost of fuel 2

        Returns:
            Substitution economics analysis
        """
        # Calculate current margins
        fuel1_margin = fuel1_price - fuel1_cost
        fuel2_margin = fuel2_price - fuel2_cost

        # Calculate substitution volume
        substitution_volume = min(fuel1_demand, fuel2_demand) * substitution_rate

        # Calculate impact on margins
        fuel1_lost_margin = substitution_volume * fuel1_margin
        fuel2_gained_margin = substitution_volume * fuel2_margin

        # Net economic impact
        net_impact = fuel2_gained_margin - fuel1_lost_margin

        # Break-even substitution rate
        margin_difference = fuel2_margin - fuel1_margin
        breakeven_rate = abs(margin_difference) / max(fuel1_margin, fuel2_margin) if max(fuel1_margin, fuel2_margin) > 0 else 0

        return {
            'substitution_volume': substitution_volume,
            'fuel1_lost_margin': fuel1_lost_margin,
            'fuel2_gained_margin': fuel2_gained_margin,
            'net_economic_impact': net_impact,
            'profitable_substitution': net_impact > 0,
            'breakeven_substitution_rate': breakeven_rate,
            'current_substitution_rate': substitution_rate,
            'margin_improvement': net_impact / substitution_volume if substitution_volume > 0 else 0
        }
