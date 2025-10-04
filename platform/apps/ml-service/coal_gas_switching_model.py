"""
Coal-to-Gas Switching Economics Model

Models for analyzing economics of fuel switching between coal and natural gas:
- Switching cost calculations
- Break-even price analysis
- Generation efficiency comparisons
- Environmental cost considerations
- Regional switching opportunities
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class CoalGasSwitchingModel:
    """
    Coal-to-gas fuel switching economics model.

    Features:
    - Break-even price calculations
    - Generation efficiency comparisons
    - Environmental cost analysis
    - Regional switching optimization
    """

    def __init__(self):
        # Typical generation characteristics
        self.generation_specs = {
            'coal': {
                'heat_rate': 10.0,      # MMBtu/MWh
                'efficiency': 34.1,     # % (assumes subcritical)
                'variable_cost': 2.0,   # $/MWh (excluding fuel)
                'startup_cost': 100,    # $/MW
                'ramp_rate': 0.02,      # MW/min per MW capacity
                'min_runtime': 4,       # hours
                'co2_emissions': 0.92   # tonnes CO2/MWh
            },
            'gas_ccgt': {
                'heat_rate': 7.0,       # MMBtu/MWh
                'efficiency': 48.7,     # %
                'variable_cost': 1.5,   # $/MWh (excluding fuel)
                'startup_cost': 50,     # $/MW
                'ramp_rate': 0.05,      # MW/min per MW capacity
                'min_runtime': 1,       # hours
                'co2_emissions': 0.37   # tonnes CO2/MWh
            },
            'gas_simple': {
                'heat_rate': 10.5,      # MMBtu/MWh
                'efficiency': 32.5,     # %
                'variable_cost': 2.5,   # $/MWh (excluding fuel)
                'startup_cost': 30,     # $/MW
                'ramp_rate': 0.08,      # MW/min per MW capacity
                'min_runtime': 0.5,     # hours
                'co2_emissions': 0.55   # tonnes CO2/MWh
            }
        }

    def calculate_breakeven_prices(
        self,
        coal_price: float,
        gas_price: float,
        coal_plant_type: str = 'coal',
        gas_plant_type: str = 'gas_ccgt',
        carbon_price: float = 0,
        include_startup: bool = False
    ) -> Dict[str, float]:
        """
        Calculate break-even prices for fuel switching.

        Args:
            coal_price: Coal price ($/tonne)
            gas_price: Gas price ($/MMBtu)
            coal_plant_type: Type of coal plant
            gas_plant_type: Type of gas plant
            carbon_price: Carbon price ($/tonne CO2)
            include_startup: Whether to include startup costs

        Returns:
            Break-even analysis results
        """
        coal_spec = self.generation_specs[coal_plant_type]
        gas_spec = self.generation_specs[gas_plant_type]

        # Calculate fuel costs per MWh
        coal_fuel_cost = coal_price * coal_spec['heat_rate'] / 1000  # Convert tonne to MMBtu
        gas_fuel_cost = gas_price * gas_spec['heat_rate']

        # Calculate variable costs per MWh
        coal_var_cost = coal_fuel_cost + coal_spec['variable_cost']
        gas_var_cost = gas_fuel_cost + gas_spec['variable_cost']

        # Add carbon costs if applicable
        coal_carbon_cost = coal_spec['co2_emissions'] * carbon_price
        gas_carbon_cost = gas_spec['co2_emissions'] * carbon_price

        coal_total_var_cost = coal_var_cost + coal_carbon_cost
        gas_total_var_cost = gas_var_cost + gas_carbon_cost

        # Calculate break-even prices
        # Gas plant should run when: gas_total_cost < coal_total_cost

        # Current cost difference
        cost_difference = gas_total_var_cost - coal_total_var_cost

        # Break-even gas price for current coal price
        breakeven_gas_price = (coal_total_var_cost - coal_spec['variable_cost'] - coal_carbon_cost) / gas_spec['heat_rate']

        # Break-even coal price for current gas price
        breakeven_coal_price = (gas_total_var_cost - gas_spec['variable_cost'] - gas_carbon_cost) * 1000 / coal_spec['heat_rate']

        # Switching threshold
        switching_advantage = coal_total_var_cost - gas_total_var_cost

        return {
            'coal_fuel_cost_per_mwh': coal_fuel_cost,
            'gas_fuel_cost_per_mwh': gas_fuel_cost,
            'coal_total_var_cost': coal_total_var_cost,
            'gas_total_var_cost': gas_total_var_cost,
            'cost_difference': cost_difference,
            'breakeven_gas_price': breakeven_gas_price,
            'breakeven_coal_price': breakeven_coal_price,
            'switching_advantage': switching_advantage,
            'gas_advantage': switching_advantage > 0,
            'carbon_price_impact': (coal_carbon_cost - gas_carbon_cost)
        }

    def calculate_switching_costs(
        self,
        generation_capacity: float,
        switching_frequency: str = 'daily',
        include_environmental: bool = True,
        carbon_price: float = 0
    ) -> Dict[str, float]:
        """
        Calculate costs associated with fuel switching.

        Args:
            generation_capacity: Plant capacity (MW)
            switching_frequency: 'daily', 'weekly', 'monthly'
            include_environmental: Include environmental costs
            carbon_price: Carbon price for emissions

        Returns:
            Switching cost analysis
        """
        # Startup costs for switching
        coal_startup_cost = self.generation_specs['coal']['startup_cost'] * generation_capacity
        gas_startup_cost = self.generation_specs['gas_ccgt']['startup_cost'] * generation_capacity

        # Frequency multipliers
        frequency_multipliers = {
            'daily': 30,    # ~30 switches per month
            'weekly': 4,    # 4 switches per month
            'monthly': 1    # 1 switch per month
        }

        monthly_switches = frequency_multipliers.get(switching_frequency, 1)

        # Total startup costs per month
        total_startup_costs = (coal_startup_cost + gas_startup_cost) * monthly_switches

        # Environmental costs (if included)
        environmental_costs = 0
        if include_environmental:
            # Difference in CO2 emissions when switching from coal to gas
            coal_emissions = self.generation_specs['coal']['co2_emissions'] * generation_capacity * 24 * 30  # Monthly
            gas_emissions = self.generation_specs['gas_ccgt']['co2_emissions'] * generation_capacity * 24 * 30

            emissions_reduction = coal_emissions - gas_emissions
            environmental_costs = emissions_reduction * carbon_price

        # Efficiency losses during switching
        efficiency_losses = generation_capacity * 0.02 * 24 * 30 * 50  # 2% efficiency loss, $50/MWh

        total_switching_costs = total_startup_costs + environmental_costs + efficiency_losses

        return {
            'monthly_startup_costs': total_startup_costs,
            'environmental_costs': environmental_costs,
            'efficiency_losses': efficiency_losses,
            'total_switching_costs': total_switching_costs,
            'cost_per_switch': total_switching_costs / monthly_switches,
            'switching_frequency': switching_frequency,
            'capacity': generation_capacity
        }

    def analyze_regional_switching_opportunities(
        self,
        regional_coal_prices: Dict[str, float],
        regional_gas_prices: Dict[str, float],
        regional_generation_mix: Dict[str, Dict[str, float]],
        transport_costs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze fuel switching opportunities across regions.

        Args:
            regional_coal_prices: Coal prices by region ($/tonne)
            regional_gas_prices: Gas prices by region ($/MMBtu)
            regional_generation_mix: Generation capacity by fuel type
            transport_costs: Fuel transport costs by region

        Returns:
            Regional switching analysis
        """
        regions = list(regional_coal_prices.keys())

        switching_opportunities = {}

        for region in regions:
            coal_price = regional_coal_prices[region]
            gas_price = regional_gas_prices[region]

            # Calculate break-even for this region
            breakeven = self.calculate_breakeven_prices(coal_price, gas_price)

            # Get generation mix for the region
            gen_mix = regional_generation_mix.get(region, {})

            coal_capacity = gen_mix.get('coal', 0)
            gas_capacity = gen_mix.get('gas', 0)

            # Calculate potential switching volume
            # Assume 20% of coal capacity can switch to gas
            switchable_capacity = coal_capacity * 0.2

            # Calculate economic impact
            economic_impact = breakeven['switching_advantage'] * switchable_capacity * 24 * 30  # Monthly

            switching_opportunities[region] = {
                'breakeven_analysis': breakeven,
                'coal_capacity': coal_capacity,
                'gas_capacity': gas_capacity,
                'switchable_capacity': switchable_capacity,
                'monthly_economic_impact': economic_impact,
                'switching_feasible': breakeven['gas_advantage'],
                'profitability_rank': economic_impact  # For ranking regions
            }

        # Rank regions by switching profitability
        ranked_regions = sorted(
            switching_opportunities.items(),
            key=lambda x: x[1]['profitability_rank'],
            reverse=True
        )

        return {
            'regional_opportunities': switching_opportunities,
            'ranked_regions': ranked_regions,
            'total_switchable_capacity': sum(
                opp['switchable_capacity'] for opp in switching_opportunities.values()
            ),
            'total_monthly_impact': sum(
                opp['monthly_economic_impact'] for opp in switching_opportunities.values()
            )
        }

    def forecast_switching_behavior(
        self,
        coal_price_forecast: pd.Series,
        gas_price_forecast: pd.Series,
        carbon_price_forecast: Optional[pd.Series] = None,
        historical_switching_data: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Forecast fuel switching behavior based on price forecasts.

        Args:
            coal_price_forecast: Coal price forecast
            gas_price_forecast: Gas price forecast
            carbon_price_forecast: Carbon price forecast
            historical_switching_data: Historical switching patterns

        Returns:
            Switching behavior forecast
        """
        # Align forecast series
        forecast_data = pd.DataFrame({
            'coal_price': coal_price_forecast,
            'gas_price': gas_price_forecast
        })

        if carbon_price_forecast is not None:
            forecast_data['carbon_price'] = carbon_price_forecast

        forecast_data = forecast_data.dropna()

        if len(forecast_data) == 0:
            return pd.DataFrame()

        # Calculate switching economics for each period
        switching_forecast = []

        for idx, row in forecast_data.iterrows():
            coal_price = row['coal_price']
            gas_price = row['gas_price']
            carbon_price = row.get('carbon_price', 0)

            # Calculate break-even
            breakeven = self.calculate_breakeven_prices(
                coal_price, gas_price, carbon_price=carbon_price
            )

            # Determine switching probability
            if breakeven['gas_advantage']:
                # Gas is cheaper - high switching probability
                switch_prob = min(1.0, abs(breakeven['switching_advantage']) / 10)  # Scale by advantage
            else:
                # Coal is cheaper - low switching probability
                switch_prob = max(0.0, 1 - abs(breakeven['switching_advantage']) / 10)

            # Estimate switching volume (percentage of total capacity)
            switch_volume_pct = switch_prob * 0.3  # Up to 30% of capacity can switch

            switching_forecast.append({
                'date': idx,
                'coal_price': coal_price,
                'gas_price': gas_price,
                'carbon_price': carbon_price,
                'switching_advantage': breakeven['switching_advantage'],
                'switching_probability': switch_prob,
                'switching_volume_pct': switch_volume_pct,
                'gas_advantage': breakeven['gas_advantage']
            })

        return pd.DataFrame(switching_forecast)

    def calculate_environmental_impact(
        self,
        switching_volume: float,  # MW switched from coal to gas
        operating_hours: float = 24 * 30,  # Monthly operating hours
        include_upstream: bool = True
    ) -> Dict[str, float]:
        """
        Calculate environmental impact of fuel switching.

        Args:
            switching_volume: Generation capacity switched (MW)
            operating_hours: Hours of operation per period
            include_upstream: Include upstream emissions

        Returns:
            Environmental impact analysis
        """
        coal_spec = self.generation_specs['coal']
        gas_spec = self.generation_specs['gas_ccgt']

        # Calculate emissions reduction
        coal_emissions = coal_spec['co2_emissions'] * switching_volume * operating_hours
        gas_emissions = gas_spec['co2_emissions'] * switching_volume * operating_hours

        emissions_reduction = coal_emissions - gas_emissions

        # Other pollutants reduction
        so2_reduction = (coal_spec['co2_emissions'] * 0.001 - gas_spec['co2_emissions'] * 0.0001) * switching_volume * operating_hours
        nox_reduction = (coal_spec['co2_emissions'] * 0.0005 - gas_spec['co2_emissions'] * 0.0002) * switching_volume * operating_hours
        particulate_reduction = (coal_spec['co2_emissions'] * 0.0003 - gas_spec['co2_emissions'] * 0.00001) * switching_volume * operating_hours

        # Water usage reduction (coal uses more water for cooling)
        water_reduction = 500 * switching_volume * operating_hours  # gallons

        # Upstream emissions (if included)
        upstream_reduction = 0
        if include_upstream:
            # Gas has lower upstream emissions than coal
            upstream_reduction = emissions_reduction * 0.1  # 10% additional reduction

        total_emissions_reduction = emissions_reduction + upstream_reduction

        return {
            'co2_reduction_tonnes': emissions_reduction,
            'so2_reduction_tonnes': so2_reduction,
            'nox_reduction_tonnes': nox_reduction,
            'particulate_reduction_tonnes': particulate_reduction,
            'water_reduction_gallons': water_reduction,
            'total_emissions_reduction': total_emissions_reduction,
            'upstream_emissions_reduction': upstream_reduction,
            'switching_volume_mw': switching_volume,
            'operating_hours': operating_hours
        }

    def optimize_switching_strategy(
        self,
        coal_prices: pd.Series,
        gas_prices: pd.Series,
        generation_capacity: float,
        operating_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize fuel switching strategy over time.

        Args:
            coal_prices: Coal price series
            gas_prices: Gas price series
            generation_capacity: Total capacity (MW)
            operating_constraints: Operational constraints

        Returns:
            Optimal switching strategy
        """
        # Default constraints
        if operating_constraints is None:
            operating_constraints = {
                'min_runtime_hours': 4,
                'max_switches_per_day': 2,
                'ramp_rate_mw_per_min': 50,
                'startup_time_hours': 2
            }

        # Calculate daily switching economics
        daily_data = pd.DataFrame({
            'coal_price': coal_prices,
            'gas_price': gas_prices
        })

        # Calculate break-even for each day
        switching_decisions = []

        for idx, row in daily_data.iterrows():
            breakeven = self.calculate_breakeven_prices(row['coal_price'], row['gas_price'])

            # Simple switching decision rule
            if breakeven['gas_advantage']:
                # Gas is cheaper - switch to gas
                decision = 'gas'
                switch_cost = operating_constraints.get('startup_time_hours', 2) * 50  # $/MW * hours
            else:
                # Coal is cheaper - stay with coal
                decision = 'coal'
                switch_cost = 0

            switching_decisions.append({
                'date': idx,
                'decision': decision,
                'switching_advantage': breakeven['switching_advantage'],
                'switch_cost': switch_cost,
                'gas_advantage': breakeven['gas_advantage']
            })

        decisions_df = pd.DataFrame(switching_decisions)

        # Calculate total costs and benefits
        total_switch_costs = decisions_df['switch_cost'].sum()
        gas_operating_days = (decisions_df['decision'] == 'gas').sum()

        # Estimate fuel cost savings
        avg_gas_advantage = decisions_df[decisions_df['gas_advantage']]['switching_advantage'].mean()
        fuel_cost_savings = avg_gas_advantage * generation_capacity * gas_operating_days * 24

        net_benefit = fuel_cost_savings - total_switch_costs

        return {
            'optimal_strategy': decisions_df,
            'total_switch_costs': total_switch_costs,
            'gas_operating_days': gas_operating_days,
            'fuel_cost_savings': fuel_cost_savings,
            'net_benefit': net_benefit,
            'switching_frequency': len(decisions_df[decisions_df['decision'] == 'gas']) / len(decisions_df),
            'average_gas_advantage': avg_gas_advantage if avg_gas_advantage else 0
        }
