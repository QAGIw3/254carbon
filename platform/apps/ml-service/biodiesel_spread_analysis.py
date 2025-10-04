"""
Biodiesel/Diesel Spread Analysis Model

Model for analyzing biodiesel-diesel price spreads and economics:
- Biodiesel production economics
- Feedstock cost analysis
- Blending economics optimization
- Policy incentive valuation
- Market arbitrage opportunities
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class BiodieselSpreadAnalysis:
    """
    Biodiesel-diesel spread analysis model.

    Features:
    - Biodiesel production cost modeling
    - Feedstock economics analysis
    - Blending optimization
    - Policy incentive valuation
    """

    def __init__(self):
        # Biodiesel production parameters
        self.production_parameters = {
            'soybean_oil_yield': 0.18,     # gallons biodiesel per pound soybeans
            'canola_oil_yield': 0.22,      # gallons biodiesel per pound canola
            'used_cooking_oil_yield': 0.95, # gallons biodiesel per gallon UCO
            'tallow_yield': 0.90,          # gallons biodiesel per pound tallow

            'conversion_cost': 0.35,       # $/gallon conversion cost
            'glycerol_credit': 0.15,       # $/gallon glycerol byproduct credit
            'meal_credit': 0.05,           # $/gallon meal byproduct credit
        }

        # Policy incentives
        self.policy_incentives = {
            'federal_biodiesel_credit': 1.00,  # $/gallon BTC
            'state_incentives': 0.20,         # $/gallon average state incentives
            'lcfs_credit': 0.50,              # $/gallon LCFS credit value
            'rin_value_d4': 1.25              # $/gallon D4 RIN value
        }

    def calculate_biodiesel_production_costs(
        self,
        feedstock_type: str,
        feedstock_price: float,
        natural_gas_price: float = 3.50,
        electricity_price: float = 0.08
    ) -> Dict[str, float]:
        """
        Calculate biodiesel production costs from feedstocks.

        Args:
            feedstock_type: Type of feedstock ('soybean_oil', 'canola_oil', 'uco', 'tallow')
            feedstock_price: Feedstock price ($/lb for oils, $/gallon for UCO)
            natural_gas_price: Natural gas price ($/MMBtu)
            electricity_price: Electricity price ($/kWh)

        Returns:
            Production cost analysis
        """
        # Get production parameters for feedstock
        if feedstock_type == 'soybean_oil':
            yield_factor = self.production_parameters['soybean_oil_yield']
            meal_credit = self.production_parameters['meal_credit']
        elif feedstock_type == 'canola_oil':
            yield_factor = self.production_parameters['canola_oil_yield']
            meal_credit = self.production_parameters['meal_credit'] * 0.8  # Lower meal value
        elif feedstock_type == 'uco':
            yield_factor = self.production_parameters['used_cooking_oil_yield']
            meal_credit = 0  # No meal byproduct
        elif feedstock_type == 'tallow':
            yield_factor = self.production_parameters['tallow_yield']
            meal_credit = 0  # No meal byproduct
        else:
            raise ValueError(f"Unknown feedstock type: {feedstock_type}")

        # Calculate feedstock cost per gallon biodiesel
        feedstock_cost_per_gallon = feedstock_price / yield_factor

        # Calculate energy costs
        # Assume 0.8 MMBtu natural gas and 0.5 kWh electricity per gallon
        energy_cost = (natural_gas_price * 0.8) + (electricity_price * 0.5)

        # Total variable costs
        total_variable_cost = (
            feedstock_cost_per_gallon +
            self.production_parameters['conversion_cost'] +
            energy_cost
        )

        # Subtract byproduct credits
        net_production_cost = total_variable_cost - self.production_parameters['glycerol_credit'] - meal_credit

        return {
            'feedstock_cost_per_gallon': feedstock_cost_per_gallon,
            'energy_cost_per_gallon': energy_cost,
            'conversion_cost_per_gallon': self.production_parameters['conversion_cost'],
            'glycerol_credit_per_gallon': self.production_parameters['glycerol_credit'],
            'meal_credit_per_gallon': meal_credit,
            'total_variable_cost': total_variable_cost,
            'net_production_cost': net_production_cost,
            'feedstock_type': feedstock_type,
            'feedstock_price': feedstock_price,
            'yield_factor': yield_factor
        }

    def analyze_biodiesel_diesel_spread(
        self,
        biodiesel_prices: pd.Series,
        diesel_prices: pd.Series,
        biodiesel_yield: float = 1.0,
        rin_value: float = 1.25,
        btc_value: float = 1.00
    ) -> Dict[str, Any]:
        """
        Analyze biodiesel-diesel price spread economics.

        Args:
            biodiesel_prices: Biodiesel price series
            diesel_prices: Diesel price series
            biodiesel_yield: Biodiesel yield from feedstock
            rin_value: D4 RIN value ($/RIN)
            btc_value: Biodiesel tax credit value ($/gallon)

        Returns:
            Spread analysis results
        """
        # Align data
        data = pd.DataFrame({
            'biodiesel': biodiesel_prices,
            'diesel': diesel_prices
        }).dropna()

        if len(data) < 20:
            return {'error': 'Insufficient data for spread analysis'}

        # Calculate spreads
        data['gross_spread'] = data['biodiesel'] - data['diesel']
        data['net_spread'] = data['gross_spread'] + rin_value + btc_value

        # Calculate biodiesel equivalent value
        # Biodiesel trades at diesel price + incentives - production costs
        data['biodiesel_equivalent'] = data['diesel'] + rin_value + btc_value

        # Analyze spread statistics
        spread_stats = {
            'mean_gross_spread': data['gross_spread'].mean(),
            'std_gross_spread': data['gross_spread'].std(),
            'mean_net_spread': data['net_spread'].mean(),
            'correlation': np.corrcoef(data['biodiesel'], data['diesel'])[0, 1],
            'spread_volatility': data['gross_spread'].std()
        }

        # Identify arbitrage opportunities
        arbitrage_threshold = 0.10  # $0.10/gallon minimum spread for arbitrage

        arbitrage_periods = data[data['net_spread'] > arbitrage_threshold]

        return {
            'spread_statistics': spread_stats,
            'arbitrage_opportunities': len(arbitrage_periods),
            'max_arbitrage_spread': data['net_spread'].max(),
            'average_arbitrage_spread': data['net_spread'][data['net_spread'] > arbitrage_threshold].mean(),
            'biodiesel_equivalent_avg': data['biodiesel_equivalent'].mean(),
            'data_points': len(data),
            'analysis_period': f'{data.index.min().date()} to {data.index.max().date()}'
        }

    def optimize_biodiesel_blend_economics(
        self,
        diesel_price: float,
        biodiesel_price: float,
        rin_value: float = 1.25,
        btc_value: float = 1.00,
        max_blend_rate: float = 0.20,  # 20% biodiesel blend limit
        production_cost: float = 0.50
    ) -> Dict[str, Any]:
        """
        Optimize biodiesel blending economics.

        Args:
            diesel_price: Diesel price ($/gallon)
            biodiesel_price: Biodiesel price ($/gallon)
            rin_value: D4 RIN value ($/RIN)
            btc_value: Biodiesel tax credit ($/gallon)
            max_blend_rate: Maximum biodiesel blend rate
            production_cost: Biodiesel production cost ($/gallon)

        Returns:
            Blend optimization results
        """
        # Calculate economics at different blend rates
        blend_rates = np.linspace(0, max_blend_rate, 21)  # 0% to 20% in 1% increments

        blend_economics = []

        for blend_rate in blend_rates:
            # Calculate blended fuel cost
            blended_cost = (diesel_price * (1 - blend_rate)) + (biodiesel_price * blend_rate)

            # Calculate RIN generation (1.5 RINs per gallon biodiesel)
            rin_generation = blend_rate * 1.5

            # Calculate total value including incentives
            rin_value_total = rin_generation * rin_value
            btc_value_total = blend_rate * btc_value

            total_incentive_value = rin_value_total + btc_value_total

            # Net cost after incentives
            net_cost = blended_cost - total_incentive_value

            # Profitability
            profitability = diesel_price - net_cost

            blend_economics.append({
                'blend_rate': blend_rate,
                'blended_cost': blended_cost,
                'rin_generation': rin_generation,
                'btc_value': btc_value_total,
                'total_incentives': total_incentive_value,
                'net_cost': net_cost,
                'profitability': profitability,
                'margin_per_gallon': profitability
            })

        # Find optimal blend rate (maximum profitability)
        optimal_blend = max(blend_economics, key=lambda x: x['profitability'])

        # Calculate total economics for optimal blend
        optimal_volume = 1000000  # 1 million gallons/day
        daily_rin_generation = optimal_blend['rin_generation'] * optimal_volume
        daily_profit = optimal_blend['profitability'] * optimal_volume

        return {
            'blend_economics': blend_economics,
            'optimal_blend': optimal_blend,
            'max_profitability': optimal_blend['profitability'],
            'optimal_blend_rate': optimal_blend['blend_rate'],
            'daily_rin_generation': daily_rin_generation,
            'daily_profit': daily_profit,
            'diesel_price': diesel_price,
            'biodiesel_price': biodiesel_price,
            'incentive_values': {
                'rin_value': rin_value,
                'btc_value': btc_value
            }
        }

    def forecast_feedstock_price_impact(
        self,
        feedstock_prices: Dict[str, pd.Series],
        biodiesel_demand: pd.Series,
        price_elasticity: float = -0.3
    ) -> Dict[str, pd.Series]:
        """
        Forecast impact of feedstock price changes on biodiesel economics.

        Args:
            feedstock_prices: Feedstock price data by type
            biodiesel_demand: Biodiesel demand data
            price_elasticity: Biodiesel price elasticity

        Returns:
            Feedstock price impact forecasts
        """
        impact_forecasts = {}

        for feedstock, price_series in feedstock_prices.items():
            # Align data
            combined_data = pd.DataFrame({
                'feedstock_price': price_series,
                'biodiesel_demand': biodiesel_demand
            }).dropna()

            if len(combined_data) < 20:
                continue

            # Calculate biodiesel production costs
            production_costs = []

            for _, row in combined_data.iterrows():
                # Calculate production cost for this feedstock price
                cost_analysis = self.calculate_biodiesel_production_costs(
                    feedstock, row['feedstock_price']
                )
                production_costs.append(cost_analysis['net_production_cost'])

            # Analyze relationship between feedstock prices and biodiesel demand
            feedstock_demand_corr = np.corrcoef(
                combined_data['feedstock_price'],
                combined_data['biodiesel_demand']
            )[0, 1]

            # Forecast impact on biodiesel prices and demand
            # Simple model: biodiesel price = feedstock cost + margin
            avg_feedstock_price = combined_data['feedstock_price'].mean()
            avg_biodiesel_demand = combined_data['biodiesel_demand'].mean()

            # Calculate typical margin
            # In production: Use more sophisticated margin analysis
            typical_margin = 0.30  # $/gallon

            # Forecast series
            forecast_dates = pd.date_range(
                price_series.index[-1] + pd.Timedelta(days=1),
                periods=30
            )

            # Simple forecast based on recent trends
            feedstock_trend = price_series.diff().mean()
            forecast_feedstock_prices = []

            last_price = price_series.iloc[-1]
            for _ in range(30):
                last_price += feedstock_trend
                forecast_feedstock_prices.append(last_price)

            # Calculate corresponding biodiesel price impact
            biodiesel_price_impact = []
            demand_impact = []

            for feedstock_price in forecast_feedstock_prices:
                # Calculate production cost
                cost_analysis = self.calculate_biodiesel_production_costs(feedstock, feedstock_price)
                biodiesel_price = cost_analysis['net_production_cost'] + typical_margin

                # Calculate demand impact based on price elasticity
                price_change = (biodiesel_price - (cost_analysis['net_production_cost'] + typical_margin)) / (cost_analysis['net_production_cost'] + typical_margin)
                demand_change = price_elasticity * price_change
                new_demand = avg_biodiesel_demand * (1 + demand_change)

                biodiesel_price_impact.append(biodiesel_price)
                demand_impact.append(new_demand)

            impact_forecasts[feedstock] = {
                'feedstock_price_forecast': pd.Series(forecast_feedstock_prices, index=forecast_dates),
                'biodiesel_price_impact': pd.Series(biodiesel_price_impact, index=forecast_dates),
                'demand_impact': pd.Series(demand_impact, index=forecast_dates),
                'correlation': feedstock_demand_corr,
                'elasticity_used': price_elasticity
            }

        return impact_forecasts

    def calculate_policy_incentive_impact(
        self,
        biodiesel_price: float,
        diesel_price: float,
        policy_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate impact of policy changes on biodiesel economics.

        Args:
            biodiesel_price: Current biodiesel price ($/gallon)
            diesel_price: Current diesel price ($/gallon)
            policy_scenarios: Policy change scenarios

        Returns:
            Policy impact analysis
        """
        base_incentives = sum(self.policy_incentives.values())

        policy_impacts = {}

        for scenario in policy_scenarios:
            scenario_name = scenario['name']

            # Modify incentives based on scenario
            modified_incentives = self.policy_incentives.copy()

            for incentive, change in scenario.get('incentive_changes', {}).items():
                if incentive in modified_incentives:
                    modified_incentives[incentive] = max(0, modified_incentives[incentive] + change)

            # Calculate new economics
            total_incentives = sum(modified_incentives.values())

            # Biodiesel equivalent value
            biodiesel_equivalent = diesel_price + total_incentives

            # Profitability
            profitability = biodiesel_equivalent - biodiesel_price

            # RIN generation economics
            rin_generation = 1.5  # RINs per gallon biodiesel
            rin_value = modified_incentives.get('rin_value_d4', 1.25) * rin_generation

            policy_impacts[scenario_name] = {
                'modified_incentives': modified_incentives,
                'total_incentives': total_incentives,
                'biodiesel_equivalent': biodiesel_equivalent,
                'profitability': profitability,
                'rin_value': rin_value,
                'scenario_description': scenario.get('description', ''),
                'economic_viability': profitability > 0
            }

        return {
            'policy_scenarios': policy_impacts,
            'base_incentives': self.policy_incentives,
            'current_biodiesel_price': biodiesel_price,
            'current_diesel_price': diesel_price,
            'scenarios_analyzed': len(policy_scenarios)
        }

    def analyze_regional_blend_economics(
        self,
        regional_diesel_prices: Dict[str, float],
        biodiesel_prices: Dict[str, float],
        regional_blend_limits: Dict[str, float],
        transport_costs: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Analyze biodiesel blending economics across regions.

        Args:
            regional_diesel_prices: Diesel prices by region ($/gallon)
            biodiesel_prices: Biodiesel prices by region ($/gallon)
            regional_blend_limits: Maximum blend rates by region
            transport_costs: Biodiesel transport costs by region

        Returns:
            Regional blend economics analysis
        """
        if transport_costs is None:
            transport_costs = {region: 0 for region in regional_diesel_prices.keys()}

        regional_analysis = {}

        for region, diesel_price in regional_diesel_prices.items():
            biodiesel_price = biodiesel_prices.get(region, biodiesel_prices.get('national', 3.50))
            transport_cost = transport_costs.get(region, 0)
            blend_limit = regional_blend_limits.get(region, 0.20)

            # Calculate delivered biodiesel cost
            delivered_biodiesel_cost = biodiesel_price + transport_cost

            # Calculate blend economics
            max_blend_rate = min(blend_limit, 0.20)  # Cap at 20% for B20

            # Optimize blend rate for maximum profitability
            blend_rates = np.linspace(0, max_blend_rate, 11)
            blend_profits = []

            for blend_rate in blend_rates:
                # Blended fuel cost
                blended_cost = (diesel_price * (1 - blend_rate)) + (delivered_biodiesel_cost * blend_rate)

                # Incentives
                rin_generation = blend_rate * 1.5
                btc_value = blend_rate * self.policy_incentives['federal_biodiesel_credit']
                lcfs_value = blend_rate * self.policy_incentives['lcfs_credit']

                total_incentives = rin_generation * self.policy_incentives['rin_value_d4'] + btc_value + lcfs_value

                # Net cost and profitability
                net_cost = blended_cost - total_incentives
                profitability = diesel_price - net_cost

                blend_profits.append({
                    'blend_rate': blend_rate,
                    'profitability': profitability,
                    'net_cost': net_cost,
                    'incentives': total_incentives
                })

            # Find optimal blend
            optimal_blend = max(blend_profits, key=lambda x: x['profitability'])

            regional_analysis[region] = {
                'diesel_price': diesel_price,
                'biodiesel_price': biodiesel_price,
                'transport_cost': transport_cost,
                'blend_limit': blend_limit,
                'optimal_blend': optimal_blend,
                'max_profitability': optimal_blend['profitability'],
                'incentives_available': sum(self.policy_incentives.values()),
                'economic_feasibility': optimal_blend['profitability'] > 0
            }

        # Regional ranking
        profitable_regions = [
            region for region, analysis in regional_analysis.items()
            if analysis['economic_feasibility']
        ]

        return {
            'regional_analysis': regional_analysis,
            'profitable_regions': profitable_regions,
            'most_profitable_region': max(
                profitable_regions,
                key=lambda r: regional_analysis[r]['max_profitability']
            ) if profitable_regions else None,
            'least_profitable_region': min(
                regional_analysis.keys(),
                key=lambda r: regional_analysis[r]['max_profitability']
            ),
            'regions_analyzed': len(regional_diesel_prices)
        }
