"""
LNG Supply Chain Analytics and Optimization

Models for LNG flow optimization and supply chain analysis:
- LNG vessel tracking and routing optimization
- Liquefaction and regasification capacity analysis
- LNG price arbitrage between regions
- Seasonal demand forecasting for LNG markets
- Supply chain risk assessment
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


class LNGSupplyChainModel:
    """
    LNG supply chain analytics and optimization model.

    Features:
    - LNG vessel routing optimization
    - Liquefaction/regasification capacity analysis
    - Regional arbitrage opportunities
    - Seasonal demand forecasting
    """

    def __init__(self):
        # LNG market fundamentals
        self.lng_markets = {
            'us_gulf': {
                'export_capacity': 8.0,  # Bcf/d liquefaction capacity
                'export_terminals': ['Sabine Pass', 'Corpus Christi', 'Freeport', 'Cameron'],
                'transport_cost_to_europe': 0.80,  # $/MMBtu
                'transport_cost_to_asia': 1.20,
                'seasonal_demand': 'winter_heating'
            },
            'australia': {
                'export_capacity': 10.0,
                'export_terminals': ['Gorgon', 'Wheatstone', 'Ichthys', 'Prelude'],
                'transport_cost_to_japan': 0.60,
                'transport_cost_to_china': 0.50,
                'seasonal_demand': 'year_round'
            },
            'qatar': {
                'export_capacity': 10.5,
                'export_terminals': ['Ras Laffan'],
                'transport_cost_to_europe': 0.70,
                'transport_cost_to_asia': 0.90,
                'seasonal_demand': 'summer_cooling'
            },
            'europe': {
                'import_capacity': 15.0,  # Bcf/d regasification capacity
                'import_terminals': ['Zeebrugge', 'Rotterdam', 'Barcelona', 'Swindon'],
                'storage_capacity': 1200,  # Bcf
                'seasonal_demand': 'winter_heating'
            },
            'asia': {
                'import_capacity': 20.0,
                'import_terminals': ['Tokyo', 'Shanghai', 'Singapore', 'Incheon'],
                'storage_capacity': 800,
                'seasonal_demand': 'summer_cooling'
            }
        }

        # LNG vessel characteristics
        self.vessel_specs = {
            'standard': {
                'capacity': 3.5,  # Bcf per vessel
                'speed': 19.5,    # knots
                'fuel_consumption': 150,  # tonnes/day
                'charter_rate': 80000   # $/day
            },
            'qflex': {
                'capacity': 4.5,
                'speed': 19.5,
                'fuel_consumption': 180,
                'charter_rate': 100000
            },
            'qmax': {
                'capacity': 5.5,
                'speed': 19.5,
                'fuel_consumption': 220,
                'charter_rate': 120000
            }
        }

    def optimize_lng_routing(
        self,
        export_terminals: List[str],
        import_terminals: List[str],
        cargo_size: float = 3.5,  # Bcf
        vessel_speed: float = 19.5,  # knots
        fuel_price: float = 600  # $/tonne
    ) -> Dict[str, Any]:
        """
        Optimize LNG vessel routing between export and import terminals.

        Args:
            export_terminals: List of export terminal names
            import_terminals: List of import terminal names
            cargo_size: LNG cargo size (Bcf)
            vessel_speed: Vessel speed (knots)
            fuel_price: Fuel price ($/tonne)

        Returns:
            Optimal routing strategy
        """
        # Distance matrix (nautical miles) - simplified
        distances = self._calculate_distances(export_terminals, import_terminals)

        # Calculate voyage times and costs
        voyage_data = []

        for export in export_terminals:
            for import_terminal in import_terminals:
                distance = distances.get(f'{export}_to_{import_terminal}', 5000)  # Default 5000 nm

                # Calculate voyage time (hours)
                voyage_time_hours = (distance / vessel_speed) * 1.15  # Add 15% for maneuvering

                # Calculate fuel consumption
                fuel_consumption = self._calculate_fuel_consumption(
                    distance, vessel_speed, cargo_size
                )

                # Calculate total costs
                fuel_cost = fuel_consumption * fuel_price
                charter_cost = (voyage_time_hours / 24) * self.vessel_specs['standard']['charter_rate']
                total_cost = fuel_cost + charter_cost

                voyage_data.append({
                    'export_terminal': export,
                    'import_terminal': import_terminal,
                    'distance_nm': distance,
                    'voyage_time_days': voyage_time_hours / 24,
                    'fuel_consumption_tonnes': fuel_consumption,
                    'fuel_cost': fuel_cost,
                    'charter_cost': charter_cost,
                    'total_cost': total_cost,
                    'cost_per_mmbtu': total_cost / cargo_size
                })

        # Find optimal route (minimum cost)
        optimal_route = min(voyage_data, key=lambda x: x['total_cost'])

        return {
            'optimal_route': optimal_route,
            'all_routes': voyage_data,
            'average_cost_per_mmbtu': np.mean([v['cost_per_mmbtu'] for v in voyage_data]),
            'cargo_size_bcf': cargo_size,
            'vessel_speed_knots': vessel_speed
        }

    def _calculate_distances(self, exports: List[str], imports: List[str]) -> Dict[str, float]:
        """Calculate distances between terminals (simplified)."""
        # In production: Use actual port coordinates and routing algorithms

        # Simplified distance matrix (nautical miles)
        distance_matrix = {
            'Sabine_Pass_to_Zeebrugge': 4800,
            'Sabine_Pass_to_Rotterdam': 4600,
            'Sabine_Pass_to_Tokyo': 7200,
            'Sabine_Pass_to_Shanghai': 7800,
            'Gorgon_to_Tokyo': 2800,
            'Gorgon_to_Shanghai': 3200,
            'Ras_Laffan_to_Zeebrugge': 3800,
            'Ras_Laffan_to_Rotterdam': 3600,
            'Ras_Laffan_to_Tokyo': 4200,
            'Ras_Laffan_to_Shanghai': 4000
        }

        distances = {}
        for export in exports:
            for import_terminal in imports:
                key = f'{export}_to_{import_terminal}'
                distances[key] = distance_matrix.get(key, 5000)  # Default distance

        return distances

    def _calculate_fuel_consumption(
        self,
        distance_nm: float,
        speed_knots: float,
        cargo_bcf: float
    ) -> float:
        """Calculate fuel consumption for LNG voyage."""
        # Simplified fuel consumption model
        base_consumption = 150  # tonnes/day base

        # Adjust for speed (fuel consumption ~ speed^3)
        speed_factor = (speed_knots / 19.5) ** 3

        # Adjust for cargo load (heavier load = more fuel)
        load_factor = 1 + (cargo_bcf - 3.5) / 3.5 * 0.1  # 10% increase per Bcf above 3.5

        # Calculate voyage duration in days
        voyage_days = (distance_nm / speed_knots) / 24 * 1.15  # Add 15% for maneuvering

        total_consumption = base_consumption * speed_factor * load_factor * voyage_days

        return total_consumption

    def analyze_lng_arbitrage_opportunities(
        self,
        export_prices: Dict[str, float],  # $/MMBtu at export terminals
        import_prices: Dict[str, float],  # $/MMBtu at import terminals
        transport_costs: Dict[str, float],  # $/MMBtu transport costs
        liquefaction_cost: float = 0.8,   # $/MMBtu liquefaction cost
        regasification_cost: float = 0.3  # $/MMBtu regasification cost
    ) -> Dict[str, Any]:
        """
        Analyze LNG arbitrage opportunities between regions.

        Args:
            export_prices: Prices at export terminals
            import_prices: Prices at import terminals
            transport_costs: Transport costs between terminals
            liquefaction_cost: Liquefaction cost
            regasification_cost: Regasification cost

        Returns:
            Arbitrage analysis results
        """
        opportunities = []

        for export, export_price in export_prices.items():
            for import_terminal, import_price in import_prices.items():
                transport_cost = transport_costs.get(f'{export}_to_{import_terminal}', 1.0)

                # Calculate delivered cost
                delivered_cost = export_price + liquefaction_cost + transport_cost + regasification_cost

                # Calculate arbitrage profit
                arbitrage_profit = import_price - delivered_cost

                if arbitrage_profit > 0:
                    opportunities.append({
                        'export_terminal': export,
                        'import_terminal': import_terminal,
                        'export_price': export_price,
                        'import_price': import_price,
                        'delivered_cost': delivered_cost,
                        'arbitrage_profit': arbitrage_profit,
                        'transport_cost': transport_cost,
                        'profit_margin_pct': (arbitrage_profit / delivered_cost) * 100
                    })

        if not opportunities:
            return {'strategy': 'no_opportunity', 'profit': 0}

        # Find most profitable opportunity
        best_opportunity = max(opportunities, key=lambda x: x['arbitrage_profit'])

        # Calculate total market opportunity
        total_profit = sum(opp['arbitrage_profit'] for opp in opportunities)

        return {
            'strategy': 'lng_arbitrage',
            'best_opportunity': best_opportunity,
            'all_opportunities': opportunities,
            'total_arbitrage_profit': total_profit,
            'average_profit_margin': np.mean([opp['profit_margin_pct'] for opp in opportunities]),
            'export_markets': len(export_prices),
            'import_markets': len(import_prices)
        }

    def forecast_seasonal_lng_demand(
        self,
        historical_demand: pd.Series,
        weather_forecast: pd.Series,
        economic_indicators: Optional[Dict[str, pd.Series]] = None
    ) -> pd.Series:
        """
        Forecast seasonal LNG demand patterns.

        Args:
            historical_demand: Historical demand data
            weather_forecast: Temperature/hdd/cdd forecasts
            economic_indicators: Economic indicators (GDP, industrial production)

        Returns:
            Seasonal demand forecast
        """
        # Calculate historical seasonal patterns
        monthly_demand = historical_demand.groupby(historical_demand.index.month).agg(['mean', 'std'])

        # Weather sensitivity analysis
        weather_impact = self._analyze_weather_lng_demand(
            historical_demand, weather_forecast
        )

        # Generate seasonal forecast
        forecast_months = 12
        seasonal_forecast = pd.Series(index=range(1, forecast_months + 1))

        for month in range(1, forecast_months + 1):
            base_demand = monthly_demand.loc[month, 'mean'] if month in monthly_demand.index else historical_demand.mean()

            # Apply weather adjustments
            weather_adjustment = weather_impact.get('monthly_weather_impact', {}).get(month, 1.0)

            seasonal_forecast[month] = base_demand * weather_adjustment

        return seasonal_forecast

    def _analyze_weather_lng_demand(
        self,
        demand: pd.Series,
        weather_data: pd.Series
    ) -> Dict[str, float]:
        """Analyze weather impact on LNG demand."""
        # Simple correlation analysis
        correlation = np.corrcoef(demand, weather_data)[0, 1]

        # Monthly weather impact factors
        monthly_impact = {}
        for month in range(1, 13):
            month_mask = demand.index.month == month
            if month_mask.sum() > 5:
                month_demand = demand[month_mask]
                month_weather = weather_data[month_mask]

                if len(month_demand) > 0 and len(month_weather) > 0:
                    month_correlation = np.corrcoef(month_demand, month_weather)[0, 1]
                    monthly_impact[month] = 1 + month_correlation * 0.1  # Scale correlation to adjustment factor

        return {
            'overall_correlation': correlation,
            'monthly_weather_impact': monthly_impact
        }

    def optimize_lng_portfolio(
        self,
        export_capacities: Dict[str, float],
        import_demands: Dict[str, float],
        transport_costs: Dict[str, Dict[str, float]],
        contract_obligations: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize LNG export/import portfolio allocation.

        Args:
            export_capacities: Available export capacity by terminal
            import_demands: Import demand by terminal
            transport_costs: Transport costs between terminals
            contract_obligations: Contractual delivery obligations

        Returns:
            Optimal portfolio allocation
        """
        # Simple linear programming approach for portfolio optimization
        # In production: Use more sophisticated optimization algorithms

        exports = list(export_capacities.keys())
        imports = list(import_demands.keys())

        # Create cost matrix
        cost_matrix = np.zeros((len(exports), len(imports)))

        for i, export in enumerate(exports):
            for j, import_terminal in enumerate(imports):
                cost_matrix[i, j] = transport_costs.get(export, {}).get(import_terminal, 1.0)

        # Simple allocation (minimize total transport cost)
        # In practice: Use scipy.optimize.linprog or similar

        total_export_capacity = sum(export_capacities.values())
        total_import_demand = sum(import_demands.values())

        # Proportional allocation
        allocation = {}
        for export in exports:
            export_allocation = {}
            export_capacity = export_capacities[export]

            for import_terminal in imports:
                # Allocate proportionally to demand
                demand_share = import_demands[import_terminal] / total_import_demand
                allocated_volume = export_capacity * demand_share

                if allocated_volume > 0:
                    export_allocation[import_terminal] = allocated_volume

            if export_allocation:
                allocation[export] = export_allocation

        # Calculate total transport costs
        total_cost = 0
        for export, export_alloc in allocation.items():
            for import_terminal, volume in export_alloc.items():
                transport_cost = transport_costs.get(export, {}).get(import_terminal, 1.0)
                total_cost += volume * transport_cost

        return {
            'optimal_allocation': allocation,
            'total_transport_cost': total_cost,
            'total_volume_allocated': sum(
                sum(export_alloc.values()) for export_alloc in allocation.values()
            ),
            'export_utilization': {
                export: sum(alloc.values()) / export_capacities[export]
                for export, alloc in allocation.items()
            },
            'import_satisfaction': {
                import_terminal: min(1.0, sum(
                    alloc.get(import_terminal, 0) for alloc in allocation.values()
                ) / import_demands[import_terminal])
                for import_terminal in imports
            }
        }

    def assess_supply_chain_risks(
        self,
        vessel_availability: Dict[str, int],
        terminal_maintenance: Dict[str, List[Tuple[datetime, datetime]]],
        geopolitical_risks: Dict[str, float],
        weather_risks: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Assess risks in LNG supply chain.

        Args:
            vessel_availability: Available vessels by type
            terminal_maintenance: Maintenance schedules by terminal
            geopolitical_risks: Risk scores by region (0-1)
            weather_risks: Weather risk scores by route (0-1)

        Returns:
            Supply chain risk assessment
        """
        # Calculate vessel availability risk
        total_vessels = sum(vessel_availability.values())
        vessel_risk = 1 - (total_vessels / 50)  # Assume 50 vessels needed for full capacity

        # Calculate terminal maintenance risk
        current_date = datetime.now()
        maintenance_risk = 0

        for terminal, schedules in terminal_maintenance.items():
            for start_date, end_date in schedules:
                if start_date <= current_date <= end_date:
                    maintenance_risk += 0.1  # 10% risk per terminal under maintenance

        # Aggregate geopolitical risks
        geopolitical_risk = np.mean(list(geopolitical_risks.values()))

        # Aggregate weather risks
        weather_risk = np.mean(list(weather_risks.values()))

        # Overall supply chain risk score
        overall_risk = (vessel_risk + maintenance_risk + geopolitical_risk + weather_risk) / 4

        # Risk mitigation recommendations
        recommendations = []

        if vessel_risk > 0.3:
            recommendations.append("Increase vessel charter capacity")

        if maintenance_risk > 0.2:
            recommendations.append("Schedule maintenance during low-demand periods")

        if geopolitical_risk > 0.4:
            recommendations.append("Diversify supply sources")

        if weather_risk > 0.3:
            recommendations.append("Develop alternative routing strategies")

        return {
            'overall_risk_score': overall_risk,
            'vessel_availability_risk': vessel_risk,
            'terminal_maintenance_risk': maintenance_risk,
            'geopolitical_risk': geopolitical_risk,
            'weather_risk': weather_risk,
            'risk_level': 'high' if overall_risk > 0.6 else ('medium' if overall_risk > 0.3 else 'low'),
            'recommendations': recommendations,
            'vessel_count': total_vessels
        }

    def calculate_lng_project_economics(
        self,
        liquefaction_cost: float,  # $/tonne
        regasification_cost: float,  # $/tonne
        transport_cost: float,     # $/MMBtu
        lng_price: float,          # $/MMBtu
        project_life: int = 20,    # years
        capacity: float = 5.0,     # Mtpa
        discount_rate: float = 0.08  # 8%
    ) -> Dict[str, float]:
        """
        Calculate economics of LNG projects.

        Args:
            liquefaction_cost: Cost of liquefaction
            regasification_cost: Cost of regasification
            transport_cost: Transport cost
            lng_price: LNG selling price
            project_life: Project life in years
            capacity: Plant capacity (Mtpa)
            discount_rate: Discount rate

        Returns:
            Project economics analysis
        """
        # Convert units (Mtpa to MMBtu/day)
        # 1 tonne LNG ≈ 52 MMBtu, 1 Mtpa = 1000 tonnes/day
        daily_capacity_mmbtu = capacity * 1000 * 52 / 365.25

        # Calculate annual revenue
        annual_revenue = lng_price * daily_capacity_mmbtu * 365.25

        # Calculate annual costs
        liquefaction_annual_cost = liquefaction_cost * capacity * 1000000  # Convert Mtpa to tonnes
        regasification_annual_cost = regasification_cost * capacity * 1000000
        transport_annual_cost = transport_cost * daily_capacity_mmbtu * 365.25

        total_annual_costs = liquefaction_annual_cost + regasification_annual_cost + transport_annual_cost

        # Calculate annual cash flow
        annual_cash_flow = annual_revenue - total_annual_costs

        # Calculate NPV
        npv = 0
        for year in range(1, project_life + 1):
            discounted_cash_flow = annual_cash_flow / (1 + discount_rate) ** year
            npv += discounted_cash_flow

        # Calculate IRR (simplified)
        # In practice: Use more sophisticated IRR calculation
        irr = discount_rate  # Placeholder

        # Calculate payback period
        cumulative_cash_flow = 0
        payback_period = 0

        for year in range(1, project_life + 1):
            cumulative_cash_flow += annual_cash_flow / (1 + discount_rate) ** year
            if cumulative_cash_flow > 0 and payback_period == 0:
                payback_period = year
            elif cumulative_cash_flow > 0:
                break

        return {
            'npv': npv,
            'irr': irr,
            'annual_cash_flow': annual_cash_flow,
            'annual_revenue': annual_revenue,
            'total_annual_costs': total_annual_costs,
            'payback_period': payback_period,
            'daily_capacity_mmbtu': daily_capacity_mmbtu,
            'capacity_mtpa': capacity,
            'project_life': project_life
        }
