"""
Coal Transportation Costs and Logistics Model

Models for coal transportation economics and logistics:
- Freight rate calculations for different routes
- Vessel chartering economics
- Port congestion and demurrage costs
- Rail and truck transportation costs
- Seasonal transportation cost variations
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CoalTransportationModel:
    """
    Coal transportation economics and logistics model.

    Features:
    - Freight rate modeling for coal routes
    - Vessel chartering optimization
    - Port congestion analysis
    - Multi-modal transport cost optimization
    """

    def __init__(self):
        # Coal transportation fundamentals
        self.transport_modes = {
            'capesize': {
                'capacity_tonnes': 180000,
                'daily_rate_usd': 25000,
                'fuel_consumption': 80,  # tonnes/day
                'speed_knots': 14.5,
                'loading_rate': 50000,  # tonnes/day
                'discharge_rate': 40000  # tonnes/day
            },
            'panamax': {
                'capacity_tonnes': 75000,
                'daily_rate_usd': 18000,
                'fuel_consumption': 45,
                'speed_knots': 14.0,
                'loading_rate': 30000,
                'discharge_rate': 25000
            },
            'handymax': {
                'capacity_tonnes': 50000,
                'daily_rate_usd': 15000,
                'fuel_consumption': 35,
                'speed_knots': 13.5,
                'loading_rate': 20000,
                'discharge_rate': 18000
            }
        }

        # Major coal ports and routes
        self.coal_routes = {
            'newcastle_to_rotterdam': {
                'distance_nm': 9200,
                'typical_vessel': 'capesize',
                'seasonal_variation': 0.15,  # 15% seasonal variation
                'congestion_factor': 1.1    # 10% congestion premium
            },
            'richards_bay_to_rotterdam': {
                'distance_nm': 6800,
                'typical_vessel': 'panamax',
                'seasonal_variation': 0.12,
                'congestion_factor': 1.05
            },
            'hay_point_to_japan': {
                'distance_nm': 3800,
                'typical_vessel': 'panamax',
                'seasonal_variation': 0.10,
                'congestion_factor': 1.08
            },
            'qinhuangdao_to_shanghai': {
                'distance_nm': 800,
                'typical_vessel': 'handymax',
                'seasonal_variation': 0.08,
                'congestion_factor': 1.15
            }
        }

        # Rail and truck transport costs (simplified)
        self.land_transport_costs = {
            'us_rail': 0.03,      # $/tonne-mile
            'china_rail': 0.05,   # $/tonne-mile
            'truck_short_haul': 0.15,  # $/tonne-mile for <100 miles
            'truck_long_haul': 0.08    # $/tonne-mile for >100 miles
        }

    def calculate_sea_freight_rates(
        self,
        route: str,
        cargo_size: float,
        vessel_type: Optional[str] = None,
        fuel_price: float = 600,  # $/tonne
        include_congestion: bool = True
    ) -> Dict[str, float]:
        """
        Calculate sea freight rates for coal transportation.

        Args:
            route: Route name (e.g., 'newcastle_to_rotterdam')
            cargo_size: Cargo size in tonnes
            vessel_type: Type of vessel to use
            fuel_price: Fuel price ($/tonne)
            include_congestion: Include congestion premiums

        Returns:
            Freight rate analysis
        """
        if route not in self.coal_routes:
            raise ValueError(f"Route {route} not found")

        route_info = self.coal_routes[route]

        if vessel_type is None:
            vessel_type = route_info['typical_vessel']

        vessel = self.transport_modes[vessel_type]
        distance = route_info['distance_nm']

        # Calculate voyage time
        voyage_time_days = (distance / vessel['speed_knots']) / 24

        # Calculate fuel consumption
        fuel_consumption = vessel['fuel_consumption'] * voyage_time_days

        # Calculate fuel cost
        fuel_cost = fuel_consumption * fuel_price

        # Calculate charter cost
        charter_cost = vessel['daily_rate_usd'] * voyage_time_days

        # Calculate port costs (loading and discharge)
        loading_time_days = cargo_size / vessel['loading_rate']
        discharge_time_days = cargo_size / vessel['discharge_rate']
        port_time_days = loading_time_days + discharge_time_days

        port_cost = vessel['daily_rate_usd'] * port_time_days * 0.5  # 50% of daily rate for port time

        # Base freight rate
        base_freight_rate = (fuel_cost + charter_cost + port_cost) / cargo_size

        # Add seasonal variation
        seasonal_factor = 1 + route_info['seasonal_variation'] * np.sin(2 * np.pi * datetime.now().month / 12)

        # Add congestion premium
        congestion_factor = route_info['congestion_factor'] if include_congestion else 1.0

        # Final freight rate
        final_freight_rate = base_freight_rate * seasonal_factor * congestion_factor

        return {
            'freight_rate_usd_per_tonne': final_freight_rate,
            'base_rate': base_freight_rate,
            'seasonal_adjustment': seasonal_factor,
            'congestion_premium': congestion_factor,
            'fuel_cost': fuel_cost,
            'charter_cost': charter_cost,
            'port_cost': port_cost,
            'voyage_time_days': voyage_time_days,
            'fuel_consumption_tonnes': fuel_consumption,
            'cargo_size_tonnes': cargo_size,
            'vessel_type': vessel_type
        }

    def optimize_vessel_chartering(
        self,
        cargo_requirements: List[Dict[str, Any]],
        available_vessels: Dict[str, Dict[str, Any]],
        charter_period_days: int = 365
    ) -> Dict[str, Any]:
        """
        Optimize vessel chartering strategy for coal transport.

        Args:
            cargo_requirements: List of cargo requirements with routes and sizes
            available_vessels: Available vessels by type and rate
            charter_period_days: Charter period in days

        Returns:
            Optimal chartering strategy
        """
        # Simple optimization for vessel selection
        # In production: Use more sophisticated algorithms

        total_cargo_volume = sum(cargo['size'] for cargo in cargo_requirements)

        # Select vessels based on cargo size and economics
        selected_vessels = []
        remaining_cargo = total_cargo_volume

        # Sort vessels by cost efficiency (rate per tonne capacity)
        vessel_efficiency = {}
        for vessel_type, vessel_info in available_vessels.items():
            efficiency = vessel_info['daily_rate'] / self.transport_modes[vessel_type]['capacity_tonnes']
            vessel_efficiency[vessel_type] = efficiency

        # Sort by efficiency (lowest cost per tonne first)
        sorted_vessels = sorted(vessel_efficiency.items(), key=lambda x: x[1])

        for vessel_type, _ in sorted_vessels:
            vessel = self.transport_modes[vessel_type]
            vessel_capacity = vessel['capacity_tonnes']

            # Calculate how many vessels of this type we need
            vessels_needed = int(np.ceil(remaining_cargo / vessel_capacity))

            for _ in range(vessels_needed):
                if remaining_cargo <= 0:
                    break

                vessel_cargo = min(vessel_capacity, remaining_cargo)
                charter_cost = vessel['daily_rate_usd'] * charter_period_days

                selected_vessels.append({
                    'vessel_type': vessel_type,
                    'capacity_tonnes': vessel_capacity,
                    'cargo_assigned': vessel_cargo,
                    'charter_cost': charter_cost,
                    'cost_per_tonne': charter_cost / vessel_cargo,
                    'utilization_rate': vessel_cargo / vessel_capacity
                })

                remaining_cargo -= vessel_cargo

        # Calculate total costs
        total_charter_cost = sum(v['charter_cost'] for v in selected_vessels)
        average_cost_per_tonne = total_charter_cost / total_cargo_volume if total_cargo_volume > 0 else 0

        return {
            'selected_vessels': selected_vessels,
            'total_charter_cost': total_charter_cost,
            'total_cargo_volume': total_cargo_volume,
            'average_cost_per_tonne': average_cost_per_tonne,
            'vessels_utilized': len(selected_vessels),
            'remaining_cargo': remaining_cargo,
            'charter_period_days': charter_period_days
        }

    def calculate_port_congestion_costs(
        self,
        port_name: str,
        vessel_waiting_time_days: float,
        demurrage_rate: float = 25000,  # $/day
        cargo_value: float = 100  # $/tonne
    ) -> Dict[str, float]:
        """
        Calculate costs associated with port congestion.

        Args:
            port_name: Name of the port
            vessel_waiting_time_days: Average waiting time
            demurrage_rate: Demurrage rate ($/day)
            cargo_value: Value of cargo ($/tonne)

        Returns:
            Congestion cost analysis
        """
        # Demurrage costs
        demurrage_cost = vessel_waiting_time_days * demurrage_rate

        # Opportunity cost of delayed cargo
        # Assume cargo could be sold immediately if not delayed
        opportunity_cost = vessel_waiting_time_days * cargo_value * 0.001  # Small opportunity cost factor

        # Total congestion costs
        total_congestion_cost = demurrage_cost + opportunity_cost

        # Cost per tonne
        # Assume standard vessel size for calculation
        assumed_cargo_size = 75000  # tonnes (Panamax)
        cost_per_tonne = total_congestion_cost / assumed_cargo_size

        return {
            'demurrage_cost': demurrage_cost,
            'opportunity_cost': opportunity_cost,
            'total_congestion_cost': total_congestion_cost,
            'cost_per_tonne': cost_per_tonne,
            'vessel_waiting_time_days': vessel_waiting_time_days,
            'demurrage_rate': demurrage_rate,
            'assumed_cargo_size': assumed_cargo_size,
            'port_name': port_name
        }

    def optimize_multi_modal_transport(
        self,
        origin: str,
        destination: str,
        cargo_size: float,
        transport_options: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Optimize transportation across multiple modes (sea, rail, truck).

        Args:
            origin: Origin location
            destination: Destination location
            cargo_size: Cargo size in tonnes
            transport_options: Transport options with costs and times

        Returns:
            Optimal transport strategy
        """
        # Evaluate each transport option
        transport_analysis = []

        for mode, specs in transport_options.items():
            cost_per_tonne = specs['cost'] / cargo_size
            total_cost = specs['cost']
            transit_time = specs['time_days']

            transport_analysis.append({
                'mode': mode,
                'total_cost': total_cost,
                'cost_per_tonne': cost_per_tonne,
                'transit_time_days': transit_time,
                'cost_per_tonne_per_day': cost_per_tonne / transit_time if transit_time > 0 else 0
            })

        # Find optimal option (minimize cost per tonne per day)
        optimal_option = min(transport_analysis, key=lambda x: x['cost_per_tonne_per_day'])

        # Calculate total transport costs for cargo
        total_transport_cost = optimal_option['total_cost']

        return {
            'optimal_transport': optimal_option,
            'all_options': transport_analysis,
            'total_transport_cost': total_transport_cost,
            'origin': origin,
            'destination': destination,
            'cargo_size_tonnes': cargo_size,
            'cost_savings_vs_alternatives': sum(
                opt['total_cost'] - optimal_option['total_cost']
                for opt in transport_analysis if opt['mode'] != optimal_option['mode']
            )
        }

    def forecast_seasonal_transport_costs(
        self,
        route: str,
        historical_costs: pd.Series,
        lookback_years: int = 3
    ) -> pd.Series:
        """
        Forecast seasonal transportation cost patterns.

        Args:
            route: Transportation route
            historical_costs: Historical cost data
            lookback_years: Years of historical data

        Returns:
            Seasonal cost forecast
        """
        # Calculate monthly average costs
        monthly_costs = historical_costs.groupby(historical_costs.index.month).agg(['mean', 'std'])

        # Calculate seasonal volatility
        monthly_volatility = monthly_costs['std'] / monthly_costs['mean']

        # Generate seasonal forecast
        forecast_months = 12
        seasonal_forecast = pd.Series(index=range(1, forecast_months + 1))

        for month in range(1, forecast_months + 1):
            if month in monthly_costs.index:
                base_cost = monthly_costs.loc[month, 'mean']

                # Add trend (assume 3% annual increase)
                trend_factor = 1 + 0.03 * (lookback_years / 2)

                # Add seasonal variation based on volatility
                seasonal_variation = np.random.normal(0, monthly_volatility.get(month, 0.1))

                forecast_cost = base_cost * trend_factor * (1 + seasonal_variation)
                seasonal_forecast[month] = max(0, forecast_cost)  # Ensure non-negative
            else:
                # Use overall average if no historical data
                seasonal_forecast[month] = historical_costs.mean()

        return seasonal_forecast

    def calculate_carbon_transport_costs(
        self,
        route: str,
        cargo_size: float,
        carbon_price: float = 50,  # $/tonne CO2
        fuel_consumption_factor: float = 0.003  # tonnes CO2 per tonne-mile
    ) -> Dict[str, float]:
        """
        Calculate carbon costs for coal transportation.

        Args:
            route: Transportation route
            cargo_size: Cargo size in tonnes
            carbon_price: Carbon price ($/tonne CO2)
            fuel_consumption_factor: CO2 emissions per tonne-mile

        Returns:
            Carbon cost analysis
        """
        if route not in self.coal_routes:
            raise ValueError(f"Route {route} not found")

        distance = self.coal_routes[route]['distance_nm']

        # Convert nautical miles to statute miles (simplified)
        distance_miles = distance * 1.15

        # Calculate total tonne-miles
        tonne_miles = cargo_size * distance_miles

        # Calculate CO2 emissions
        co2_emissions = tonne_miles * fuel_consumption_factor

        # Calculate carbon costs
        carbon_cost = co2_emissions * carbon_price

        # Cost per tonne of cargo
        carbon_cost_per_tonne = carbon_cost / cargo_size

        return {
            'co2_emissions_tonnes': co2_emissions,
            'carbon_cost': carbon_cost,
            'carbon_cost_per_tonne': carbon_cost_per_tonne,
            'distance_miles': distance_miles,
            'tonne_miles': tonne_miles,
            'carbon_price': carbon_price,
            'emission_factor': fuel_consumption_factor,
            'route': route,
            'cargo_size_tonnes': cargo_size
        }

    def analyze_transport_efficiency(
        self,
        vessel_utilization: pd.Series,
        fuel_efficiency: pd.Series,
        cargo_throughput: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze transportation efficiency metrics.

        Args:
            vessel_utilization: Vessel utilization rates over time
            fuel_efficiency: Fuel efficiency (tonnes per nautical mile)
            cargo_throughput: Cargo throughput (tonnes per day)

        Returns:
            Efficiency analysis
        """
        # Calculate average utilization
        avg_utilization = vessel_utilization.mean()

        # Calculate fuel efficiency trends
        fuel_efficiency_trend = fuel_efficiency.diff().mean()

        # Calculate throughput trends
        throughput_trend = cargo_throughput.diff().mean()

        # Identify efficiency improvement opportunities
        opportunities = []

        if avg_utilization < 0.8:
            opportunities.append("Improve vessel utilization through better scheduling")

        if fuel_efficiency_trend > 0:
            opportunities.append("Implement fuel efficiency measures")

        if throughput_trend < 0:
            opportunities.append("Optimize port operations and loading rates")

        return {
            'average_utilization': avg_utilization,
            'fuel_efficiency_trend': fuel_efficiency_trend,
            'throughput_trend': throughput_trend,
            'efficiency_score': (avg_utilization + (1 - abs(fuel_efficiency_trend)) + (1 - abs(throughput_trend))) / 3,
            'improvement_opportunities': opportunities,
            'data_points': len(vessel_utilization)
        }
