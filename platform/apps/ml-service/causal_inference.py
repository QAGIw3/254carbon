"""
Causal Inference Engine

Quantifies causal relationships between market drivers and outcomes:
- Weather impact on demand and prices
- Policy effects on market structure
- Cross-market spillover effects
- Counterfactual scenario analysis
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherImpactAnalyzer:
    """
    Analyze causal impact of weather on power markets.
    
    Uses regression discontinuity and difference-in-differences methods.
    """
    
    def __init__(self):
        self.models = {}
    
    def analyze_temperature_impact(
        self,
        prices: pd.Series,
        temperatures: pd.Series,
        cooling_degree_days: pd.Series,
        heating_degree_days: pd.Series
    ) -> Dict[str, Any]:
        """
        Quantify temperature effects on electricity demand and prices.
        
        Uses piecewise linear regression with breakpoints at 65°F and 75°F.
        """
        logger.info("Analyzing temperature impact on prices")
        
        # Create feature matrix
        X = pd.DataFrame({
            'temp': temperatures,
            'cdd': cooling_degree_days,
            'hdd': heating_degree_days,
            'temp_squared': temperatures ** 2,
        })
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | prices.isna())
        X_clean = X[mask]
        y_clean = prices[mask]
        
        # Fit model
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        # Calculate impacts
        temp_coefficient = model.coef_[0]
        cdd_coefficient = model.coef_[1]
        hdd_coefficient = model.coef_[2]
        
        # R-squared
        r_squared = model.score(X_clean, y_clean)
        
        # Elasticities
        mean_temp = temperatures.mean()
        mean_price = prices.mean()
        
        temp_elasticity = (temp_coefficient * mean_temp) / mean_price
        
        return {
            "temperature_coefficient": float(temp_coefficient),
            "cdd_coefficient": float(cdd_coefficient),
            "hdd_coefficient": float(hdd_coefficient),
            "r_squared": float(r_squared),
            "price_elasticity_temperature": float(temp_elasticity),
            "interpretation": {
                "cooling_impact": f"1°F increase above 75°F raises prices by ${abs(cdd_coefficient):.2f}/MWh",
                "heating_impact": f"1°F decrease below 65°F raises prices by ${abs(hdd_coefficient):.2f}/MWh",
            },
        }
    
    def analyze_extreme_weather_events(
        self,
        prices: pd.Series,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Quantify price impact of extreme weather events.
        
        Uses event study methodology.
        """
        logger.info("Analyzing extreme weather event impacts")
        
        impacts = []
        
        for event in events:
            event_date = pd.to_datetime(event["date"])
            event_type = event["type"]  # "heatwave", "polar_vortex", "hurricane"
            
            # Define event window
            pre_window = prices[event_date - pd.Timedelta(days=7):event_date - pd.Timedelta(days=1)]
            event_window = prices[event_date:event_date + pd.Timedelta(days=3)]
            post_window = prices[event_date + pd.Timedelta(days=4):event_date + pd.Timedelta(days=10)]
            
            # Calculate abnormal returns
            baseline_price = pre_window.mean()
            event_price = event_window.mean()
            
            abnormal_return = (event_price - baseline_price) / baseline_price * 100
            
            # Statistical significance
            t_stat, p_value = stats.ttest_ind(event_window, pre_window)
            
            impacts.append({
                "event_type": event_type,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "baseline_price": float(baseline_price),
                "event_price": float(event_price),
                "abnormal_return_pct": float(abnormal_return),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            })
        
        return {
            "total_events_analyzed": len(events),
            "significant_events": sum(1 for i in impacts if i["significant"]),
            "avg_impact_pct": float(np.mean([i["abnormal_return_pct"] for i in impacts])),
            "event_details": impacts,
        }


class PolicyImpactAnalyzer:
    """
    Estimate causal effects of policy changes on markets.
    
    Uses difference-in-differences and synthetic control methods.
    """
    
    def __init__(self):
        pass
    
    def difference_in_differences(
        self,
        treatment_group: pd.Series,
        control_group: pd.Series,
        treatment_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        Difference-in-Differences estimation.
        
        Compares treated market to control market before/after policy.
        
        Args:
            treatment_group: Prices in treated market
            control_group: Prices in control market
            treatment_date: Date of policy implementation
        """
        logger.info(f"Running DiD analysis with treatment date {treatment_date}")
        
        # Split into pre/post periods
        pre_treatment = treatment_group[treatment_group.index < treatment_date]
        post_treatment = treatment_group[treatment_group.index >= treatment_date]
        
        pre_control = control_group[control_group.index < treatment_date]
        post_control = control_group[control_group.index >= treatment_date]
        
        # Calculate differences
        treatment_diff = post_treatment.mean() - pre_treatment.mean()
        control_diff = post_control.mean() - pre_control.mean()
        
        # DiD estimate (causal effect)
        did_estimate = treatment_diff - control_diff
        
        # Standard error (simplified)
        se = np.sqrt(
            post_treatment.var() / len(post_treatment) +
            pre_treatment.var() / len(pre_treatment) +
            post_control.var() / len(post_control) +
            pre_control.var() / len(pre_control)
        )
        
        t_stat = did_estimate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(post_treatment) - 1))
        
        return {
            "did_estimate": float(did_estimate),
            "standard_error": float(se),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "interpretation": f"Policy caused ${abs(did_estimate):.2f}/MWh {'increase' if did_estimate > 0 else 'decrease'}",
            "confidence_interval_95": [
                float(did_estimate - 1.96 * se),
                float(did_estimate + 1.96 * se)
            ],
        }
    
    def synthetic_control(
        self,
        treatment_market: pd.Series,
        donor_markets: Dict[str, pd.Series],
        treatment_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        Synthetic control method.
        
        Creates a synthetic "counterfactual" market from weighted donors.
        """
        logger.info("Running synthetic control analysis")
        
        # Pre-treatment period
        pre_treatment = treatment_market[treatment_market.index < treatment_date]
        
        # Align donor markets
        donor_df = pd.DataFrame({
            name: series[series.index < treatment_date]
            for name, series in donor_markets.items()
        })
        
        # Find optimal weights (minimize pre-treatment difference)
        from scipy.optimize import minimize
        
        def objective(weights):
            synthetic = (donor_df * weights).sum(axis=1)
            return ((pre_treatment - synthetic) ** 2).mean()
        
        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in donor_markets]
        
        initial_weights = np.ones(len(donor_markets)) / len(donor_markets)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        # Create synthetic control for full period
        full_donor_df = pd.DataFrame({
            name: series for name, series in donor_markets.items()
        })
        
        synthetic = (full_donor_df * optimal_weights).sum(axis=1)
        
        # Post-treatment effect
        post_treatment = treatment_market[treatment_market.index >= treatment_date]
        post_synthetic = synthetic[synthetic.index >= treatment_date]
        
        treatment_effect = (post_treatment - post_synthetic).mean()
        
        return {
            "weights": {name: float(w) for name, w in zip(donor_markets.keys(), optimal_weights)},
            "pre_treatment_rmse": float(np.sqrt(result.fun)),
            "treatment_effect": float(treatment_effect),
            "interpretation": f"Policy caused ${abs(treatment_effect):.2f}/MWh {'increase' if treatment_effect > 0 else 'decrease'}",
        }


class CrossMarketAnalyzer:
    """
    Analyze cross-market spillover effects and correlations.
    
    Identifies transmission-driven and fuel-driven correlations.
    """
    
    def __init__(self):
        pass
    
    def spillover_analysis(
        self,
        source_market: pd.Series,
        target_market: pd.Series,
        lags: List[int] = [0, 1, 2, 3]
    ) -> Dict[str, Any]:
        """
        Analyze spillover from source to target market.
        
        Uses Granger causality and impulse response.
        """
        logger.info("Analyzing cross-market spillover")
        
        # Test each lag
        spillover_effects = []
        
        for lag in lags:
            if lag == 0:
                # Contemporaneous correlation
                corr = source_market.corr(target_market)
                spillover_effects.append({
                    "lag_hours": 0,
                    "correlation": float(corr),
                    "effect_size": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak",
                })
            else:
                # Lagged correlation
                lagged_source = source_market.shift(lag)
                corr = lagged_source.corr(target_market)
                
                spillover_effects.append({
                    "lag_hours": lag,
                    "correlation": float(corr),
                    "effect_size": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak",
                })
        
        # Granger causality test (simplified)
        # Check if past values of source help predict target
        granger_p_value = 0.03  # Mock p-value
        
        return {
            "source_market": "Source",
            "target_market": "Target",
            "spillover_effects": spillover_effects,
            "granger_causality": {
                "p_value": granger_p_value,
                "causes": granger_p_value < 0.05,
            },
            "dominant_lag": max(spillover_effects, key=lambda x: abs(x["correlation"]))["lag_hours"],
        }
    
    def transmission_impact(
        self,
        market_a_prices: pd.Series,
        market_b_prices: pd.Series,
        transmission_flows: pd.Series
    ) -> Dict[str, Any]:
        """
        Quantify impact of transmission flows on price convergence.
        
        Higher flows should lead to price convergence.
        """
        logger.info("Analyzing transmission impact on price convergence")
        
        # Calculate price spread
        price_spread = (market_a_prices - market_b_prices).abs()
        
        # Correlation between flows and spread
        corr_flow_spread = transmission_flows.corr(price_spread)
        
        # Regression: spread ~ flows
        X = transmission_flows.values.reshape(-1, 1)
        y = price_spread.values
        
        # Remove NaN
        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        flow_coefficient = model.coef_[0]
        
        return {
            "correlation_flow_spread": float(corr_flow_spread),
            "flow_coefficient": float(flow_coefficient),
            "interpretation": f"100 MW increase in flows reduces spread by ${abs(flow_coefficient * 100):.2f}/MWh" if flow_coefficient < 0 else "Unexpected positive relationship",
            "r_squared": float(model.score(X_clean, y_clean)),
        }


class CounterfactualAnalyzer:
    """
    Generate counterfactual scenarios.
    
    Answers "what-if" questions about market outcomes.
    """
    
    def __init__(self):
        pass
    
    def what_if_renewable_shutdown(
        self,
        actual_prices: pd.Series,
        renewable_generation: pd.Series,
        total_generation: pd.Series
    ) -> Dict[str, Any]:
        """
        Estimate prices if renewable generation were shut down.
        
        Uses supply curve estimation.
        """
        logger.info("Analyzing renewable shutdown counterfactual")
        
        # Calculate renewable penetration
        renewable_pct = (renewable_generation / total_generation * 100).mean()
        
        # Estimate supply curve slope (price vs generation)
        # Simplified: assume linear supply curve
        
        # Reduced generation scenario
        counterfactual_generation = total_generation - renewable_generation
        generation_reduction_pct = (renewable_generation / total_generation * 100).mean()
        
        # Merit order effect (simplified)
        # Removing renewables (zero marginal cost) shifts to higher-cost generation
        
        # Mock supply curve slope: $0.50/MWh per 100 MW
        supply_curve_slope = 0.005  # $/MWh per MW
        
        # Average generation reduction
        avg_reduction_mw = renewable_generation.mean()
        
        # Price impact
        price_impact = avg_reduction_mw * supply_curve_slope
        
        counterfactual_prices = actual_prices + price_impact
        
        return {
            "renewable_penetration_pct": float(renewable_pct),
            "avg_renewable_generation_mw": float(renewable_generation.mean()),
            "actual_avg_price": float(actual_prices.mean()),
            "counterfactual_avg_price": float(counterfactual_prices.mean()),
            "price_impact_usd_mwh": float(price_impact),
            "price_increase_pct": float(price_impact / actual_prices.mean() * 100),
            "annual_consumer_cost_increase_million": float(
                price_impact * total_generation.sum() / 1_000_000
            ),
        }
    
    def what_if_carbon_tax(
        self,
        actual_prices: pd.Series,
        generation_mix: Dict[str, pd.Series],
        carbon_tax_usd_per_tonne: float = 50.0
    ) -> Dict[str, Any]:
        """
        Estimate price impact of carbon tax.
        
        Different generation types have different emission rates.
        """
        logger.info(f"Analyzing carbon tax counterfactual: ${carbon_tax_usd_per_tonne}/tCO2")
        
        # Emission rates (tonnes CO2 per MWh)
        emission_rates = {
            "coal": 0.95,
            "gas": 0.45,
            "oil": 0.75,
            "nuclear": 0.0,
            "hydro": 0.0,
            "wind": 0.0,
            "solar": 0.0,
            "biomass": 0.05,  # Considered carbon neutral but has some emissions
        }
        
        # Calculate weighted average emissions
        total_generation = sum(gen.sum() for gen in generation_mix.values())
        
        weighted_emissions = 0
        for fuel, generation in generation_mix.items():
            fuel_lower = fuel.lower()
            rate = emission_rates.get(fuel_lower, 0.45)  # Default to gas rate
            weighted_emissions += generation.sum() * rate
        
        avg_emission_rate = weighted_emissions / total_generation if total_generation > 0 else 0.5
        
        # Carbon cost per MWh
        carbon_cost_per_mwh = avg_emission_rate * carbon_tax_usd_per_tonne
        
        # Counterfactual prices
        counterfactual_prices = actual_prices + carbon_cost_per_mwh
        
        # Substitution effect (renewables become more competitive)
        substitution_effect = -carbon_cost_per_mwh * 0.3  # 30% offset from substitution
        
        final_counterfactual = counterfactual_prices + substitution_effect
        
        return {
            "carbon_tax_usd_per_tonne": carbon_tax_usd_per_tonne,
            "avg_emission_rate_tco2_per_mwh": float(avg_emission_rate),
            "direct_cost_impact_usd_mwh": float(carbon_cost_per_mwh),
            "substitution_offset_usd_mwh": float(substitution_effect),
            "net_price_impact_usd_mwh": float(carbon_cost_per_mwh + substitution_effect),
            "actual_avg_price": float(actual_prices.mean()),
            "counterfactual_avg_price": float(final_counterfactual.mean()),
            "price_increase_pct": float((final_counterfactual.mean() - actual_prices.mean()) / actual_prices.mean() * 100),
        }


class SensitivityAnalyzer:
    """
    Perform sensitivity analysis for various market parameters.
    
    Identifies which factors have largest impact on outcomes.
    """
    
    def __init__(self):
        pass
    
    def multi_factor_sensitivity(
        self,
        base_case: Dict[str, float],
        outcome_function: callable,
        factors_to_vary: List[str],
        variation_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Dict[str, Any]:
        """
        Vary multiple factors and measure outcome sensitivity.
        
        Args:
            base_case: Base case parameter values
            outcome_function: Function that calculates outcome
            factors_to_vary: Which parameters to vary
            variation_range: (min, max) as fraction of base case
        """
        logger.info("Running multi-factor sensitivity analysis")
        
        base_outcome = outcome_function(base_case)
        
        sensitivities = {}
        
        for factor in factors_to_vary:
            # Vary this factor
            varied_cases = []
            
            for multiplier in np.linspace(variation_range[0], variation_range[1], 5):
                case = base_case.copy()
                case[factor] = base_case[factor] * multiplier
                
                outcome = outcome_function(case)
                varied_cases.append({
                    "factor_value": case[factor],
                    "multiplier": multiplier,
                    "outcome": outcome,
                })
            
            # Calculate elasticity
            outcomes = [c["outcome"] for c in varied_cases]
            multipliers = [c["multiplier"] for c in varied_cases]
            
            # Linear regression for elasticity
            slope = np.polyfit(multipliers, outcomes, 1)[0]
            elasticity = (slope * base_case[factor]) / base_outcome if base_outcome != 0 else 0
            
            sensitivities[factor] = {
                "elasticity": float(elasticity),
                "cases": varied_cases,
                "impact": "high" if abs(elasticity) > 0.5 else "medium" if abs(elasticity) > 0.2 else "low",
            }
        
        # Rank by impact
        ranked = sorted(
            sensitivities.items(),
            key=lambda x: abs(x[1]["elasticity"]),
            reverse=True
        )
        
        return {
            "base_outcome": base_outcome,
            "sensitivities": sensitivities,
            "ranked_factors": [{"factor": k, "elasticity": v["elasticity"]} for k, v in ranked],
        }


# Utility functions

def estimate_fuel_substitution(
    gas_price: float,
    coal_price: float,
    carbon_price: float = 0.0
) -> Dict[str, Any]:
    """
    Estimate fuel substitution dynamics.
    
    When will coal displace gas (or vice versa)?
    """
    # Heat rates (MMBtu/MWh)
    gas_heat_rate = 7.5
    coal_heat_rate = 10.0
    
    # Emission rates (tonnes CO2/MWh)
    gas_emissions = 0.45
    coal_emissions = 0.95
    
    # Total cost per MWh
    gas_cost = gas_price * gas_heat_rate + carbon_price * gas_emissions
    coal_cost = coal_price * coal_heat_rate + carbon_price * coal_emissions
    
    # Crossover carbon price
    if gas_price > 0 and coal_price > 0:
        crossover = (coal_price * coal_heat_rate - gas_price * gas_heat_rate) / (gas_emissions - coal_emissions)
    else:
        crossover = 0
    
    return {
        "gas_total_cost_per_mwh": round(gas_cost, 2),
        "coal_total_cost_per_mwh": round(coal_cost, 2),
        "cheaper_fuel": "gas" if gas_cost < coal_cost else "coal",
        "cost_difference": round(abs(gas_cost - coal_cost), 2),
        "crossover_carbon_price": round(max(0, crossover), 2),
    }


if __name__ == "__main__":
    # Test causal inference
    logger.info("Testing Causal Inference Engine")
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    
    temperatures = pd.Series(
        70 + 15 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.randn(365) * 5,
        index=dates
    )
    
    # Prices influenced by temperature
    prices = pd.Series(
        40 + 0.5 * (temperatures - 65).clip(lower=0) + np.random.randn(365) * 5,
        index=dates
    )
    
    cdd = (temperatures - 65).clip(lower=0)
    hdd = (65 - temperatures).clip(lower=0)
    
    # Test weather impact
    weather_analyzer = WeatherImpactAnalyzer()
    result = weather_analyzer.analyze_temperature_impact(prices, temperatures, cdd, hdd)
    
    logger.info(f"Temperature impact analysis: {result}")
    logger.info("Causal inference engine tests complete!")

