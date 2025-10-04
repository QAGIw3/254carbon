"""
Carbon Intensity Calculations Model

Model for calculating carbon intensity of various fuels and pathways:
- Lifecycle carbon intensity analysis
- Feedstock-specific emissions factors
- Transportation and distribution emissions
- Land use change impact assessment
- Policy compliance calculations
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class CarbonIntensityCalculator:
    """
    Carbon intensity calculation and analysis model.

    Features:
    - Lifecycle emissions analysis
    - Feedstock-specific calculations
    - Transportation impact modeling
    - Policy compliance assessment
    """

    def __init__(self):
        # Carbon intensity factors (gCO2e/MJ)
        self.carbon_intensity_factors = {
            'gasoline': {
                'extraction': 15.0,
                'refining': 8.0,
                'transport': 2.0,
                'combustion': 75.0,
                'total': 100.0
            },
            'diesel': {
                'extraction': 18.0,
                'refining': 10.0,
                'transport': 2.5,
                'combustion': 75.0,
                'total': 105.5
            },
            'ethanol_corn': {
                'farming': 25.0,
                'processing': 15.0,
                'transport': 3.0,
                'combustion': 75.0,
                'total': 118.0
            },
            'ethanol_sugarcane': {
                'farming': 8.0,
                'processing': 12.0,
                'transport': 3.0,
                'combustion': 75.0,
                'total': 98.0
            },
            'biodiesel_soy': {
                'farming': 20.0,
                'processing': 25.0,
                'transport': 3.0,
                'combustion': 75.0,
                'total': 123.0
            },
            'biodiesel_canola': {
                'farming': 18.0,
                'processing': 22.0,
                'transport': 3.0,
                'combustion': 75.0,
                'total': 118.0
            },
            'biodiesel_uco': {
                'farming': 0.0,      # Waste feedstock
                'processing': 20.0,
                'transport': 3.0,
                'combustion': 75.0,
                'total': 98.0
            },
            'biodiesel_tallow': {
                'farming': 0.0,      # Waste feedstock
                'processing': 18.0,
                'transport': 3.0,
                'combustion': 75.0,
                'total': 96.0
            },
            'natural_gas': {
                'extraction': 12.0,
                'processing': 3.0,
                'transport': 1.5,
                'combustion': 56.0,
                'total': 72.5
            },
            'lng': {
                'extraction': 12.0,
                'liquefaction': 8.0,
                'transport': 4.0,
                'regasification': 2.0,
                'combustion': 56.0,
                'total': 82.0
            },
            'coal': {
                'mining': 8.0,
                'transport': 2.0,
                'combustion': 95.0,
                'total': 105.0
            },
            'nuclear': {
                'mining': 2.0,
                'enrichment': 3.0,
                'transport': 1.0,
                'combustion': 0.0,  # No combustion emissions
                'total': 6.0
            },
            'solar': {
                'manufacturing': 25.0,
                'installation': 5.0,
                'operation': 0.0,
                'decommissioning': 2.0,
                'total': 32.0
            },
            'wind': {
                'manufacturing': 15.0,
                'installation': 3.0,
                'operation': 0.0,
                'decommissioning': 1.0,
                'total': 19.0
            }
        }

        # Land use change factors (gCO2e/MJ)
        self.land_use_factors = {
            'corn_ethanol': 25.0,      # Corn ethanol land use change
            'soy_biodiesel': 20.0,     # Soy biodiesel land use change
            'sugarcane_ethanol': 5.0,  # Low land use change for sugarcane
            'palm_biodiesel': 45.0,    # High land use change for palm
            'waste_feedstocks': 0.0    # No land use change for waste
        }

    def calculate_fuel_carbon_intensity(
        self,
        fuel_type: str,
        pathway: str = 'conventional',
        include_land_use: bool = True,
        transport_distance: float = 1000,  # km
        transport_mode: str = 'truck'
    ) -> Dict[str, float]:
        """
        Calculate carbon intensity for a specific fuel and pathway.

        Args:
            fuel_type: Type of fuel
            pathway: Production pathway
            include_land_use: Include land use change emissions
            transport_distance: Transport distance in km
            transport_mode: Transport mode ('truck', 'rail', 'ship')

        Returns:
            Carbon intensity analysis
        """
        if fuel_type not in self.carbon_intensity_factors:
            raise ValueError(f"Carbon intensity data not available for {fuel_type}")

        base_ci = self.carbon_intensity_factors[fuel_type]

        # Calculate transport emissions (gCO2e/MJ)
        transport_emissions = self._calculate_transport_emissions(
            transport_distance, transport_mode, fuel_type
        )

        # Calculate land use change emissions
        land_use_emissions = 0
        if include_land_use and pathway in self.land_use_factors:
            land_use_emissions = self.land_use_factors[pathway]

        # Total carbon intensity
        total_ci = base_ci['total'] + transport_emissions + land_use_emissions

        return {
            'total_carbon_intensity': total_ci,
            'base_emissions': base_ci['total'],
            'transport_emissions': transport_emissions,
            'land_use_emissions': land_use_emissions,
            'fuel_type': fuel_type,
            'pathway': pathway,
            'transport_distance': transport_distance,
            'transport_mode': transport_mode,
            'ci_per_mj': total_ci,
            'ci_per_gallon': total_ci * self._get_energy_density(fuel_type) if fuel_type != 'electric' else 0
        }

    def _calculate_transport_emissions(
        self,
        distance: float,
        mode: str,
        fuel_type: str
    ) -> float:
        """Calculate transport emissions for fuel delivery."""
        # Transport emission factors (gCO2e/tonne-km)
        transport_factors = {
            'truck': 100,    # gCO2e/tonne-km
            'rail': 25,      # gCO2e/tonne-km
            'ship': 15,      # gCO2e/tonne-km
            'pipeline': 5    # gCO2e/tonne-km
        }

        emission_factor = transport_factors.get(mode, 50)

        # Assume average fuel density for transport calculations
        fuel_density = self._get_fuel_density(fuel_type)

        # Calculate emissions per MJ of fuel
        emissions_per_tonne_km = emission_factor
        emissions_per_mj = emissions_per_tonne_km / (fuel_density * 1000) * distance / 1000

        return emissions_per_mj

    def _get_energy_density(self, fuel_type: str) -> float:
        """Get energy density in MJ/L for different fuels."""
        energy_densities = {
            'gasoline': 32.0,
            'diesel': 35.8,
            'ethanol': 21.1,
            'biodiesel': 32.9,
            'natural_gas': 0.036,  # MJ/L (gaseous)
            'lng': 25.0,           # MJ/L (liquid)
            'jet_fuel': 35.0,
            'fuel_oil': 40.0
        }

        return energy_densities.get(fuel_type, 35.0)

    def _get_fuel_density(self, fuel_type: str) -> float:
        """Get fuel density in kg/L for transport calculations."""
        densities = {
            'gasoline': 0.75,
            'diesel': 0.85,
            'ethanol': 0.79,
            'biodiesel': 0.88,
            'natural_gas': 0.0007,  # kg/L (gaseous)
            'lng': 0.45,            # kg/L (liquid)
            'jet_fuel': 0.80,
            'fuel_oil': 0.95
        }

        return densities.get(fuel_type, 0.85)

    def compare_fuel_carbon_intensities(
        self,
        fuel_types: List[str],
        pathways: Optional[Dict[str, str]] = None,
        include_transport: bool = True
    ) -> pd.DataFrame:
        """
        Compare carbon intensities across multiple fuels.

        Args:
            fuel_types: List of fuel types to compare
            pathways: Production pathways for each fuel
            include_transport: Include transport emissions

        Returns:
            Carbon intensity comparison DataFrame
        """
        if pathways is None:
            pathways = {fuel: 'conventional' for fuel in fuel_types}

        comparison_data = []

        for fuel_type in fuel_types:
            pathway = pathways.get(fuel_type, 'conventional')

            ci_analysis = self.calculate_fuel_carbon_intensity(
                fuel_type, pathway, include_land_use=True, include_transport=include_transport
            )

            comparison_data.append({
                'fuel_type': fuel_type,
                'pathway': pathway,
                'carbon_intensity_gco2e_mj': ci_analysis['total_carbon_intensity'],
                'base_emissions': ci_analysis['base_emissions'],
                'transport_emissions': ci_analysis['transport_emissions'],
                'land_use_emissions': ci_analysis['land_use_emissions'],
                'energy_density_mj_l': self._get_energy_density(fuel_type)
            })

        return pd.DataFrame(comparison_data)

    def calculate_blend_carbon_intensity(
        self,
        blend_components: Dict[str, float],
        component_cis: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate carbon intensity of fuel blends.

        Args:
            blend_components: Volume fractions of each component
            component_cis: Carbon intensity of each component (gCO2e/MJ)

        Returns:
            Blend carbon intensity analysis
        """
        # Normalize blend components to sum to 1.0
        total_fraction = sum(blend_components.values())
        if total_fraction == 0:
            raise ValueError("Blend components cannot sum to zero")

        normalized_components = {
            component: fraction / total_fraction
            for component, fraction in blend_components.items()
        }

        # Calculate weighted average carbon intensity
        weighted_ci = sum(
            normalized_components[component] * component_cis.get(component, 0)
            for component in normalized_components.keys()
        )

        # Calculate energy-weighted carbon intensity
        # Different fuels have different energy densities
        energy_weighted_ci = 0
        total_energy = 0

        for component, fraction in normalized_components.items():
            energy_density = self._get_energy_density(component)
            component_energy = fraction * energy_density
            total_energy += component_energy

            component_ci = component_cis.get(component, 0)
            energy_weighted_ci += component_energy * component_ci

        if total_energy > 0:
            energy_weighted_ci /= total_energy

        return {
            'volume_weighted_ci': weighted_ci,
            'energy_weighted_ci': energy_weighted_ci,
            'blend_components': normalized_components,
            'component_cis': component_cis,
            'total_energy_density': total_energy,
            'blend_method': 'energy_weighted' if energy_weighted_ci > 0 else 'volume_weighted'
        }

    def forecast_carbon_intensity_trends(
        self,
        historical_ci_data: pd.Series,
        technology_improvements: Dict[str, float] = None,
        policy_targets: Optional[Dict[str, float]] = None,
        forecast_horizon: int = 10
    ) -> pd.Series:
        """
        Forecast future carbon intensity trends.

        Args:
            historical_ci_data: Historical carbon intensity data
            technology_improvements: Annual improvement rates by technology
            policy_targets: Policy-driven reduction targets
            forecast_horizon: Years to forecast

        Returns:
            Carbon intensity trend forecast
        """
        if technology_improvements is None:
            technology_improvements = {
                'efficiency': 0.02,    # 2% annual efficiency improvement
                'renewables': 0.03,    # 3% annual renewable penetration
                'ccs': 0.01            # 1% annual CCS deployment
            }

        # Calculate historical trend
        ci_trend = historical_ci_data.diff().mean()
        ci_volatility = historical_ci_data.std()

        # Forecast future carbon intensity
        years = np.arange(forecast_horizon)
        forecast_values = []

        current_ci = historical_ci_data.iloc[-1]

        for year in years:
            # Apply technology improvements
            efficiency_reduction = technology_improvements['efficiency']
            renewables_reduction = technology_improvements['renewables']
            ccs_reduction = technology_improvements['ccs']

            total_reduction = efficiency_reduction + renewables_reduction + ccs_reduction

            # Apply reduction (compounded)
            reduction_factor = 1 - total_reduction
            current_ci *= reduction_factor

            # Add some variability
            random_variation = np.random.normal(0, ci_volatility * 0.1)
            current_ci += random_variation

            forecast_values.append(max(0, current_ci))  # Ensure non-negative

        # Create forecast series
        last_date = historical_ci_data.index[-1]
        forecast_dates = pd.date_range(
            last_date + pd.Timedelta(days=365),
            periods=forecast_horizon,
            freq='Y'
        )

        return pd.Series(forecast_values, index=forecast_dates)

    def analyze_policy_compliance(
        self,
        fuel_cis: Dict[str, float],
        policy_thresholds: Dict[str, float],
        compliance_period: str = 'annual'
    ) -> Dict[str, Any]:
        """
        Analyze compliance with carbon intensity policies.

        Args:
            fuel_cis: Carbon intensities by fuel type
            policy_thresholds: Policy thresholds by fuel/policy type
            compliance_period: Compliance period ('annual', 'quarterly')

        Returns:
            Policy compliance analysis
        """
        compliance_analysis = {}

        for fuel_type, ci in fuel_cis.items():
            fuel_compliance = {}

            for policy, threshold in policy_thresholds.items():
                if ci <= threshold:
                    compliance_status = 'compliant'
                    margin = threshold - ci
                else:
                    compliance_status = 'non_compliant'
                    margin = ci - threshold

                fuel_compliance[policy] = {
                    'carbon_intensity': ci,
                    'threshold': threshold,
                    'compliance_status': compliance_status,
                    'margin': margin,
                    'margin_percent': (margin / threshold) * 100
                }

            compliance_analysis[fuel_type] = fuel_compliance

        # Overall compliance summary
        total_fuels = len(fuel_cis)
        compliant_fuels = sum(
            1 for fuel_analysis in compliance_analysis.values()
            for policy_analysis in fuel_analysis.values()
            if policy_analysis['compliance_status'] == 'compliant'
        )

        compliance_rate = compliant_fuels / total_fuels if total_fuels > 0 else 0

        return {
            'compliance_analysis': compliance_analysis,
            'overall_compliance_rate': compliance_rate,
            'compliant_fuels': sum(
                1 for fuel_analysis in compliance_analysis.values()
                if any(policy['compliance_status'] == 'compliant' for policy in fuel_analysis.values())
            ),
            'policy_thresholds': policy_thresholds,
            'compliance_period': compliance_period
        }

    def calculate_lifecycle_emissions(
        self,
        fuel_pathway: str,
        production_location: str = 'us_average',
        end_use: str = 'transportation'
    ) -> Dict[str, float]:
        """
        Calculate detailed lifecycle emissions for a fuel pathway.

        Args:
            fuel_pathway: Fuel production pathway
            production_location: Production location
            end_use: End use application

        Returns:
            Lifecycle emissions breakdown
        """
        # Simplified lifecycle analysis
        # In production: Use detailed LCA models

        base_emissions = self.carbon_intensity_factors.get(fuel_pathway, {})

        if not base_emissions:
            return {'error': f'No emissions data for pathway: {fuel_pathway}'}

        # Location-specific adjustments
        location_factors = {
            'us_average': 1.0,
            'california': 0.95,    # Lower emissions due to regulations
            'texas': 1.05,         # Higher emissions due to intensive production
            'midwest': 1.02,       # Moderate emissions
            'northeast': 0.98      # Lower emissions due to cleaner grid
        }

        location_factor = location_factors.get(production_location, 1.0)

        # End-use adjustments
        end_use_factors = {
            'transportation': 1.0,
            'electricity': 0.95,   # More efficient end use
            'heating': 1.05,       # Less efficient end use
            'industrial': 1.02     # Moderate efficiency
        }

        end_use_factor = end_use_factors.get(end_use, 1.0)

        # Calculate adjusted emissions
        adjusted_emissions = {}
        for stage, emissions in base_emissions.items():
            adjusted_emissions[stage] = emissions * location_factor * end_use_factor

        total_adjusted = sum(adjusted_emissions.values())

        return {
            'lifecycle_emissions': adjusted_emissions,
            'total_carbon_intensity': total_adjusted,
            'location_factor': location_factor,
            'end_use_factor': end_use_factor,
            'fuel_pathway': fuel_pathway,
            'production_location': production_location,
            'end_use': end_use
        }

    def optimize_fuel_mix_carbon_intensity(
        self,
        fuel_mix: Dict[str, float],
        target_ci: float,
        optimization_constraint: str = 'cost'
    ) -> Dict[str, Any]:
        """
        Optimize fuel mix to achieve target carbon intensity.

        Args:
            fuel_mix: Current fuel mix (volume fractions)
            target_ci: Target carbon intensity (gCO2e/MJ)
            optimization_constraint: 'cost', 'volume', or 'availability'

        Returns:
            Optimized fuel mix
        """
        # Get carbon intensities for all fuels in mix
        fuel_cis = {}
        for fuel in fuel_mix.keys():
            if fuel in self.carbon_intensity_factors:
                fuel_cis[fuel] = self.carbon_intensity_factors[fuel]['total']

        if not fuel_cis:
            return {'error': 'No carbon intensity data for fuel mix'}

        # Simple optimization (could be enhanced with linear programming)
        current_weighted_ci = sum(
            fuel_mix[fuel] * fuel_cis.get(fuel, 0)
            for fuel in fuel_mix.keys()
        )

        # Calculate required adjustments
        ci_reduction_needed = current_weighted_ci - target_ci

        if ci_reduction_needed <= 0:
            return {
                'optimized_mix': fuel_mix,
                'achieved_ci': current_weighted_ci,
                'target_achieved': True,
                'no_changes_needed': True
            }

        # Identify lowest and highest CI fuels
        sorted_fuels = sorted(fuel_cis.items(), key=lambda x: x[1])

        # Increase low-CI fuels, decrease high-CI fuels
        optimized_mix = fuel_mix.copy()

        # Increase lowest CI fuel
        lowest_ci_fuel = sorted_fuels[0][0]
        if lowest_ci_fuel in optimized_mix:
            # Increase by 10% (or available capacity)
            increase_amount = min(0.1, 1 - sum(optimized_mix.values()))
            optimized_mix[lowest_ci_fuel] += increase_amount

        # Decrease highest CI fuel
        highest_ci_fuel = sorted_fuels[-1][0]
        if highest_ci_fuel in optimized_mix:
            decrease_amount = min(increase_amount, optimized_mix[highest_ci_fuel])
            optimized_mix[highest_ci_fuel] -= decrease_amount

        # Normalize mix
        total_mix = sum(optimized_mix.values())
        if total_mix > 0:
            optimized_mix = {k: v / total_mix for k, v in optimized_mix.items()}

        # Calculate new carbon intensity
        new_weighted_ci = sum(
            optimized_mix[fuel] * fuel_cis.get(fuel, 0)
            for fuel in optimized_mix.keys()
        )

        return {
            'original_mix': fuel_mix,
            'optimized_mix': optimized_mix,
            'original_ci': current_weighted_ci,
            'optimized_ci': new_weighted_ci,
            'target_ci': target_ci,
            'ci_reduction_achieved': current_weighted_ci - new_weighted_ci,
            'target_achieved': new_weighted_ci <= target_ci,
            'optimization_constraint': optimization_constraint,
            'fuel_cis_used': fuel_cis
        }
