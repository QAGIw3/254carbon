"""
Decarbonization Pathway Modeling System

Advanced model for analyzing decarbonization pathways and energy transition:
- Sector-specific decarbonization modeling
- Renewable energy adoption curves
- Technology cost curves
- Policy scenario impact assessment
- Stranded asset risk analysis
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)


class DecarbonizationPathwayModel:
    """
    Decarbonization pathway analysis and modeling.

    Features:
    - Sector decarbonization modeling
    - Renewable adoption forecasting
    - Technology transition analysis
    - Policy impact assessment
    """

    def __init__(self):
        # Sector emission profiles
        self.sector_emissions = {
            'power': {
                'current_emissions': 2000,  # Mt CO2/year
                'emission_intensity': 0.4,   # t CO2/MWh
                'decarbonization_rate': 0.08,  # 8% annual reduction target
                'renewable_share': 0.25,    # Current renewable share
                'technology_options': ['solar', 'wind', 'nuclear', 'ccs', 'storage']
            },
            'transportation': {
                'current_emissions': 1500,  # Mt CO2/year
                'emission_intensity': 0.25,  # t CO2/vehicle-km
                'decarbonization_rate': 0.06,  # 6% annual reduction target
                'ev_share': 0.03,           # Current EV share
                'technology_options': ['ev', 'hydrogen', 'biofuels', 'efficiency']
            },
            'industry': {
                'current_emissions': 1200,  # Mt CO2/year
                'emission_intensity': 0.3,   # t CO2/tonne production
                'decarbonization_rate': 0.05,  # 5% annual reduction target
                'ccs_share': 0.02,          # Current CCS share
                'technology_options': ['ccs', 'electrification', 'hydrogen', 'efficiency']
            },
            'buildings': {
                'current_emissions': 800,   # Mt CO2/year
                'emission_intensity': 0.15,  # t CO2/m²
                'decarbonization_rate': 0.04,  # 4% annual reduction target
                'renewable_heat_share': 0.10,  # Current renewable heat share
                'technology_options': ['heat_pumps', 'solar_thermal', 'biomass', 'efficiency']
            }
        }

        # Technology cost curves (simplified)
        self.technology_costs = {
            'solar': {
                'current_cost': 40,     # $/MWh
                'learning_rate': 0.20,  # 20% cost reduction per doubling
                'deployment_rate': 0.15  # 15% annual deployment growth
            },
            'wind': {
                'current_cost': 35,     # $/MWh
                'learning_rate': 0.15,  # 15% cost reduction per doubling
                'deployment_rate': 0.12  # 12% annual deployment growth
            },
            'nuclear': {
                'current_cost': 80,     # $/MWh
                'learning_rate': 0.05,  # 5% cost reduction per doubling
                'deployment_rate': 0.03  # 3% annual deployment growth
            },
            'ccs': {
                'current_cost': 60,     # $/MWh
                'learning_rate': 0.12,  # 12% cost reduction per doubling
                'deployment_rate': 0.08  # 8% annual deployment growth
            },
            'battery_storage': {
                'current_cost': 150,    # $/MWh
                'learning_rate': 0.18,  # 18% cost reduction per doubling
                'deployment_rate': 0.25  # 25% annual deployment growth
            }
        }

    def model_sector_decarbonization(
        self,
        sector: str,
        target_year: int = 2050,
        policy_scenario: str = 'ambitious',
        technology_mix: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Model decarbonization pathway for a specific sector.

        Args:
            sector: Economic sector to model
            target_year: Target year for net-zero
            policy_scenario: Policy ambition level
            technology_mix: Technology deployment mix

        Returns:
            Decarbonization pathway analysis
        """
        if sector not in self.sector_emissions:
            raise ValueError(f"Sector {sector} not found")

        sector_data = self.sector_emissions[sector]
        current_emissions = sector_data['current_emissions']
        target_reduction = sector_data['decarbonization_rate']

        # Adjust target based on policy scenario
        policy_multipliers = {
            'conservative': 0.8,
            'moderate': 1.0,
            'ambitious': 1.2,
            'aggressive': 1.5
        }

        adjusted_target = target_reduction * policy_multipliers.get(policy_scenario, 1.0)

        # Calculate annual emissions trajectory
        years = np.arange(datetime.now().year, target_year + 1)
        emissions_trajectory = []

        for year in years:
            years_from_now = year - datetime.now().year

            # Exponential decay with policy adjustment
            decay_factor = 1 - adjusted_target
            emissions = current_emissions * (decay_factor ** years_from_now)
            emissions_trajectory.append(emissions)

        # Technology deployment analysis
        technology_analysis = self._analyze_technology_deployment(
            sector, technology_mix, target_year
        )

        # Calculate cumulative emissions
        cumulative_emissions = sum(emissions_trajectory)

        return {
            'sector': sector,
            'emissions_trajectory': pd.Series(emissions_trajectory, index=years),
            'cumulative_emissions': cumulative_emissions,
            'target_achieved': emissions_trajectory[-1] <= current_emissions * 0.05,  # 95% reduction
            'policy_scenario': policy_scenario,
            'technology_analysis': technology_analysis,
            'current_emissions': current_emissions,
            'target_year': target_year,
            'annual_reduction_rate': adjusted_target
        }

    def _analyze_technology_deployment(
        self,
        sector: str,
        technology_mix: Optional[Dict[str, float]] = None,
        target_year: int = 2050
    ) -> Dict[str, Any]:
        """Analyze technology deployment for decarbonization."""
        if technology_mix is None:
            # Default technology mix based on sector
            if sector == 'power':
                technology_mix = {'solar': 0.4, 'wind': 0.3, 'nuclear': 0.1, 'ccs': 0.1, 'storage': 0.1}
            elif sector == 'transportation':
                technology_mix = {'ev': 0.6, 'hydrogen': 0.2, 'biofuels': 0.15, 'efficiency': 0.05}
            elif sector == 'industry':
                technology_mix = {'ccs': 0.4, 'electrification': 0.3, 'hydrogen': 0.2, 'efficiency': 0.1}
            else:
                technology_mix = {'heat_pumps': 0.4, 'solar_thermal': 0.3, 'biomass': 0.2, 'efficiency': 0.1}

        # Calculate deployment trajectories
        deployment_trajectories = {}

        for technology, share in technology_mix.items():
            if technology in self.technology_costs:
                tech_data = self.technology_costs[technology]

                # Calculate deployment over time
                years = np.arange(datetime.now().year, target_year + 1)
                deployment = []

                for year in years:
                    years_from_now = year - datetime.now().year

                    # Exponential growth with learning curve
                    cumulative_deployment = share * (1 + tech_data['deployment_rate']) ** years_from_now

                    # Apply learning curve for cost reduction
                    cost_reduction = (1 - tech_data['learning_rate']) ** years_from_now
                    current_cost = tech_data['current_cost'] * cost_reduction

                    deployment.append({
                        'year': year,
                        'deployment_share': cumulative_deployment,
                        'technology_cost': current_cost,
                        'learning_rate_applied': tech_data['learning_rate']
                    })

                deployment_trajectories[technology] = deployment

        return {
            'technology_mix': technology_mix,
            'deployment_trajectories': deployment_trajectories,
            'total_technologies': len(technology_mix),
            'dominant_technology': max(technology_mix.items(), key=lambda x: x[1])[0]
        }

    def forecast_renewable_adoption_curves(
        self,
        technology: str,
        current_capacity: float,
        policy_support: float = 1.0,
        economic_factors: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Forecast renewable energy adoption using S-curve models.

        Args:
            technology: Renewable technology type
            current_capacity: Current installed capacity (GW)
            policy_support: Policy support factor (0-2)
            economic_factors: Economic factors affecting adoption

        Returns:
            Adoption curve forecast
        """
        if economic_factors is None:
            economic_factors = {
                'cost_competitiveness': 1.0,
                'grid_integration': 1.0,
                'financing_availability': 1.0
            }

        # S-curve parameters
        max_capacity = current_capacity * 10  # Assume 10x growth potential
        adoption_rate = 0.15 + (policy_support * 0.05)  # Base 15% + policy boost

        # Calculate adoption trajectory
        years = np.arange(10)  # 10-year forecast
        adoption_curve = []

        for year in years:
            # Logistic growth curve
            t = year
            capacity = max_capacity / (1 + np.exp(-adoption_rate * (t - 5)))  # S-curve centered at year 5

            # Apply economic factors
            economic_multiplier = np.prod(list(economic_factors.values()))

            adjusted_capacity = capacity * economic_multiplier
            adoption_curve.append(adjusted_capacity)

        # Create forecast series
        forecast_years = pd.date_range(
            datetime.now() + pd.DateOffset(years=1),
            periods=10,
            freq='Y'
        )

        return pd.Series(adoption_curve, index=forecast_years)

    def analyze_stranded_asset_risk(
        self,
        asset_values: Dict[str, float],
        carbon_prices: Dict[str, pd.Series],
        asset_lifetimes: Dict[str, int],
        risk_free_rate: float = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze stranded asset risk from decarbonization.

        Args:
            asset_values: Current asset values by type
            carbon_prices: Carbon price forecasts
            asset_lifetimes: Remaining asset lifetimes (years)
            risk_free_rate: Risk-free discount rate

        Returns:
            Stranded asset risk analysis
        """
        stranded_risks = {}

        for asset_type, asset_value in asset_values.items():
            if asset_type not in asset_lifetimes:
                continue

            lifetime = asset_lifetimes[asset_type]

            # Calculate present value of carbon costs
            carbon_cost_pv = 0

            for market, price_forecast in carbon_prices.items():
                # Assume average carbon cost impact
                avg_carbon_price = price_forecast.mean()
                annual_carbon_cost = avg_carbon_price * 0.1  # 10% of carbon price as cost impact

                # Discount future costs
                for year in range(1, lifetime + 1):
                    discounted_cost = annual_carbon_cost / (1 + risk_free_rate) ** year
                    carbon_cost_pv += discounted_cost

            # Calculate stranded asset value
            stranded_value = min(asset_value, carbon_cost_pv)
            stranded_ratio = stranded_value / asset_value if asset_value > 0 else 0

            stranded_risks[asset_type] = {
                'asset_value': asset_value,
                'carbon_cost_pv': carbon_cost_pv,
                'stranded_value': stranded_value,
                'stranded_ratio': stranded_ratio,
                'remaining_lifetime': lifetime,
                'risk_level': 'high' if stranded_ratio > 0.7 else ('medium' if stranded_ratio > 0.4 else 'low')
            }

        # Portfolio-level analysis
        total_asset_value = sum(asset_values.values())
        total_stranded_value = sum(risk['stranded_value'] for risk in stranded_risks.values())

        return {
            'stranded_risks': stranded_risks,
            'portfolio_stranded_ratio': total_stranded_value / total_asset_value if total_asset_value > 0 else 0,
            'high_risk_assets': [
                asset for asset, risk in stranded_risks.items()
                if risk['risk_level'] == 'high'
            ],
            'total_asset_value': total_asset_value,
            'total_stranded_value': total_stranded_value,
            'risk_free_rate': risk_free_rate
        }

    def optimize_decarbonization_portfolio(
        self,
        available_technologies: List[str],
        investment_budget: float,
        target_reductions: Dict[str, float],
        technology_costs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize technology portfolio for decarbonization.

        Args:
            available_technologies: List of available technologies
            investment_budget: Total investment budget
            target_reductions: Emission reduction targets by sector
            technology_costs: Technology costs per unit reduction

        Returns:
            Optimal technology portfolio
        """
        if technology_costs is None:
            technology_costs = {
                'solar': 100,      # $/tonne CO2 reduced
                'wind': 80,        # $/tonne CO2 reduced
                'nuclear': 200,    # $/tonne CO2 reduced
                'ccs': 150,        # $/tonne CO2 reduced
                'storage': 120,    # $/tonne CO2 reduced
                'ev': 300,         # $/tonne CO2 reduced
                'hydrogen': 250,   # $/tonne CO2 reduced
                'biofuels': 180,   # $/tonne CO2 reduced
                'efficiency': 50   # $/tonne CO2 reduced
            }

        # Simple optimization for demonstration
        # In production: Use linear programming for optimal allocation

        # Calculate cost-effectiveness
        cost_effectiveness = {}
        for tech in available_technologies:
            if tech in technology_costs:
                cost_effectiveness[tech] = 1 / technology_costs[tech]  # Higher is better

        # Sort by cost-effectiveness
        sorted_technologies = sorted(cost_effectiveness.items(), key=lambda x: x[1], reverse=True)

        # Allocate budget to most cost-effective technologies
        portfolio_allocation = {}
        remaining_budget = investment_budget

        for technology, effectiveness in sorted_technologies:
            if remaining_budget <= 0:
                break

            # Allocate based on cost-effectiveness
            cost = technology_costs.get(technology, 100)
            max_allocation = remaining_budget

            # Check if this allocation meets any target
            allocation = min(max_allocation, remaining_budget)

            if allocation > 0:
                portfolio_allocation[technology] = allocation
                remaining_budget -= allocation

        # Calculate expected emission reductions
        total_reductions = 0
        sector_reductions = {}

        for technology, allocation in portfolio_allocation.items():
            # Assume each $100 invested reduces 1 tonne CO2
            reduction_per_dollar = 1 / technology_costs.get(technology, 100)
            reduction = allocation * reduction_per_dollar

            total_reductions += reduction

            # Allocate to sectors (simplified)
            if technology in ['solar', 'wind', 'nuclear', 'ccs', 'storage']:
                sector_reductions['power'] = sector_reductions.get('power', 0) + reduction * 0.6
            elif technology in ['ev', 'hydrogen']:
                sector_reductions['transportation'] = sector_reductions.get('transportation', 0) + reduction * 0.8
            elif technology == 'biofuels':
                sector_reductions['transportation'] = sector_reductions.get('transportation', 0) + reduction * 0.4
                sector_reductions['industry'] = sector_reductions.get('industry', 0) + reduction * 0.3
            else:
                sector_reductions['industry'] = sector_reductions.get('industry', 0) + reduction * 0.5

        return {
            'optimal_portfolio': portfolio_allocation,
            'total_investment': investment_budget - remaining_budget,
            'expected_reductions': total_reductions,
            'sector_reductions': sector_reductions,
            'cost_effectiveness_ranking': sorted_technologies,
            'budget_utilization': (investment_budget - remaining_budget) / investment_budget,
            'technologies_deployed': len(portfolio_allocation)
        }

    def assess_esg_scoring_impact(
        self,
        company_emissions: Dict[str, float],
        decarbonization_plans: Dict[str, Dict[str, Any]],
        esg_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Assess ESG scoring impact of decarbonization efforts.

        Args:
            company_emissions: Current emissions by company
            decarbonization_plans: Decarbonization plans by company
            esg_weights: ESG scoring weights

        Returns:
            ESG impact assessment
        """
        if esg_weights is None:
            esg_weights = {
                'environmental': 0.4,
                'social': 0.3,
                'governance': 0.3
            }

        esg_scores = {}

        for company, emissions in company_emissions.items():
            if company not in decarbonization_plans:
                continue

            plan = decarbonization_plans[company]

            # Environmental score (emissions reduction commitment)
            reduction_target = plan.get('reduction_target', 0.3)  # 30% default
            timeline = plan.get('timeline', 10)  # 10-year default

            # Score based on ambition and timeline
            ambition_score = min(1.0, reduction_target / 0.5)  # 50%+ reduction = full score
            timeline_score = min(1.0, timeline / 15)  # 15+ years = full score

            environmental_score = (ambition_score + timeline_score) / 2

            # Social score (simplified - job impacts, community benefits)
            social_score = 0.7  # Default moderate social impact

            # Governance score (transparency and accountability)
            governance_score = 0.8  # Default good governance

            # Weighted ESG score
            total_esg_score = (
                environmental_score * esg_weights['environmental'] +
                social_score * esg_weights['social'] +
                governance_score * esg_weights['governance']
            )

            esg_scores[company] = {
                'environmental_score': environmental_score,
                'social_score': social_score,
                'governance_score': governance_score,
                'total_esg_score': total_esg_score,
                'esg_weights': esg_weights,
                'decarbonization_plan': plan,
                'current_emissions': emissions,
                'esg_rating': 'A' if total_esg_score > 0.8 else ('B' if total_esg_score > 0.6 else 'C')
            }

        # Portfolio-level ESG analysis
        avg_esg_score = np.mean([score['total_esg_score'] for score in esg_scores.values()])

        return {
            'company_esg_scores': esg_scores,
            'portfolio_esg_score': avg_esg_score,
            'esg_rating_distribution': {
                'A': sum(1 for score in esg_scores.values() if score['esg_rating'] == 'A'),
                'B': sum(1 for score in esg_scores.values() if score['esg_rating'] == 'B'),
                'C': sum(1 for score in esg_scores.values() if score['esg_rating'] == 'C')
            },
            'companies_analyzed': len(esg_scores),
            'esg_weights_used': esg_weights
        }

    def model_energy_transition_scenarios(
        self,
        baseline_scenario: Dict[str, Any],
        transition_scenarios: List[Dict[str, Any]],
        scenario_probabilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Model multiple energy transition scenarios.

        Args:
            baseline_scenario: Baseline energy scenario
            transition_scenarios: Alternative transition scenarios
            scenario_probabilities: Scenario probabilities

        Returns:
            Energy transition scenario analysis
        """
        if scenario_probabilities is None:
            scenario_probabilities = {
                scenario['name']: 1/len(transition_scenarios) for scenario in transition_scenarios
            }

        scenario_analysis = {}

        # Analyze baseline
        scenario_analysis['baseline'] = {
            'scenario': baseline_scenario,
            'probability': 0.3,  # Baseline probability
            'expected_emissions': baseline_scenario.get('total_emissions', 5000),
            'expected_cost': baseline_scenario.get('total_cost', 1000),
            'technology_mix': baseline_scenario.get('technology_mix', {})
        }

        # Analyze transition scenarios
        for scenario in transition_scenarios:
            scenario_name = scenario['name']
            probability = scenario_probabilities.get(scenario_name, 0)

            scenario_analysis[scenario_name] = {
                'scenario': scenario,
                'probability': probability,
                'expected_emissions': scenario.get('total_emissions', 3000),
                'expected_cost': scenario.get('total_cost', 1500),
                'technology_mix': scenario.get('technology_mix', {}),
                'policy_measures': scenario.get('policy_measures', []),
                'timeline': scenario.get('timeline', 30)
            }

        # Calculate expected outcomes
        expected_emissions = sum(
            analysis['expected_emissions'] * analysis['probability']
            for analysis in scenario_analysis.values()
        )

        expected_cost = sum(
            analysis['expected_cost'] * analysis['probability']
            for analysis in scenario_analysis.values()
        )

        return {
            'scenario_analysis': scenario_analysis,
            'expected_emissions': expected_emissions,
            'expected_cost': expected_cost,
            'scenarios_analyzed': len(scenario_analysis),
            'scenario_probabilities': scenario_probabilities,
            'baseline_emissions': scenario_analysis['baseline']['expected_emissions'],
            'emission_reduction': scenario_analysis['baseline']['expected_emissions'] - expected_emissions
        }
