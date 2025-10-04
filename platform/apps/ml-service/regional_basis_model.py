"""
Regional Basis Modeling for Natural Gas Markets

Models for analyzing regional price differences (basis) in natural gas markets:
- Pipeline congestion modeling
- Storage arbitrage opportunities
- Weather-driven demand variations
- Transportation cost analysis
- Seasonal basis pattern recognition
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RegionalBasisModel:
    """
    Regional natural gas basis modeling and analysis.

    Features:
    - Basis calculation and forecasting
    - Pipeline congestion detection
    - Storage impact analysis
    - Weather-driven basis modeling
    """

    def __init__(self):
        # Regional hub relationships and transport costs
        self.hub_relationships = {
            'henry_hub': {
                'benchmark': True,
                'transport_cost_to_chicago': 0.15,  # $/MMBtu
                'transport_cost_to_nyc': 0.25,
                'transport_cost_to_california': 0.45,
                'storage_access': 'high'
            },
            'chicago': {
                'benchmark': False,
                'transport_cost_to_nyc': 0.10,
                'transport_cost_to_california': 0.30,
                'storage_access': 'medium'
            },
            'nyc': {
                'benchmark': False,
                'transport_cost_to_california': 0.20,
                'storage_access': 'high'
            },
            'california': {
                'benchmark': False,
                'storage_access': 'low'
            }
        }

        # Pipeline capacity constraints (Bcf/d)
        self.pipeline_capacities = {
            'henry_to_chicago': 8.0,
            'chicago_to_nyc': 2.5,
            'henry_to_nyc': 1.8,
            'henry_to_california': 3.2
        }

    def calculate_regional_basis(
        self,
        hub_prices: Dict[str, pd.Series],
        benchmark_hub: str = 'henry_hub'
    ) -> Dict[str, pd.Series]:
        """
        Calculate basis (price differential) relative to benchmark hub.

        Args:
            hub_prices: Dictionary of price series by hub
            benchmark_hub: Hub to use as benchmark

        Returns:
            Dictionary of basis series for each hub
        """
        if benchmark_hub not in hub_prices:
            raise ValueError(f"Benchmark hub {benchmark_hub} not found in price data")

        benchmark_prices = hub_prices[benchmark_hub]
        basis_results = {}

        for hub, prices in hub_prices.items():
            if hub == benchmark_hub:
                continue

            # Align price series
            aligned_data = pd.DataFrame({
                'benchmark': benchmark_prices,
                'hub': prices
            }).dropna()

            if len(aligned_data) == 0:
                logger.warning(f"No aligned data for {hub} vs {benchmark_hub}")
                continue

            # Calculate basis (hub price - benchmark price)
            basis = aligned_data['hub'] - aligned_data['benchmark']
            basis_results[hub] = basis

        return basis_results

    def forecast_basis_from_pipeline_flows(
        self,
        current_flows: Dict[str, float],
        pipeline_capacities: Dict[str, float],
        storage_levels: Dict[str, float],
        demand_forecast: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """
        Forecast basis using pipeline flow and storage data.

        Args:
            current_flows: Current pipeline flows (Bcf/d)
            pipeline_capacities: Pipeline capacity limits
            storage_levels: Current storage levels by region
            demand_forecast: Demand forecasts by region

        Returns:
            Basis forecasts by region
        """
        basis_forecasts = {}

        for pipeline, flow in current_flows.items():
            capacity = pipeline_capacities.get(pipeline, 0)

            if capacity == 0:
                continue

            # Calculate capacity utilization
            utilization = flow / capacity

            # High utilization typically leads to positive basis (premium)
            # Low utilization leads to negative basis (discount)

            # Simple basis model based on utilization
            # In practice: Use more sophisticated econometric models
            if utilization > 0.9:
                basis_multiplier = 1.5  # High congestion premium
            elif utilization < 0.7:
                basis_multiplier = 0.7  # Low utilization discount
            else:
                basis_multiplier = 1.0

            # Storage impact (high storage = negative basis pressure)
            regions = pipeline.split('_to_')
            if len(regions) == 2:
                from_region = regions[0]
                to_region = regions[1]

                from_storage = storage_levels.get(from_region, 0.5)  # Normalized 0-1
                to_storage = storage_levels.get(to_region, 0.5)

                # Storage differential affects basis
                storage_impact = (from_storage - to_storage) * 0.2

                basis_forecast = pd.Series([basis_multiplier + storage_impact] * 30,
                                         index=pd.date_range(datetime.now(), periods=30))

                basis_forecasts[pipeline] = basis_forecast

        return basis_forecasts

    def analyze_congestion_patterns(
        self,
        flows: pd.Series,
        capacity: float,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Analyze pipeline congestion patterns and seasonality.

        Args:
            flows: Pipeline flow series (Bcf/d)
            capacity: Pipeline capacity (Bcf/d)
            lookback_days: Days of historical data to analyze

        Returns:
            Congestion analysis results
        """
        # Calculate utilization rates
        utilization = flows / capacity

        # Identify congestion periods (utilization > 90%)
        congestion_periods = utilization > 0.9

        # Analyze congestion frequency by month
        monthly_congestion = congestion_periods.groupby(congestion_periods.index.month).mean()

        # Analyze congestion frequency by day of week
        weekly_congestion = congestion_periods.groupby(congestion_periods.index.weekday).mean()

        # Duration of congestion events
        congestion_events = self._identify_congestion_events(congestion_periods)
        avg_congestion_duration = np.mean([event['duration'] for event in congestion_events])

        return {
            'overall_utilization': utilization.mean(),
            'congestion_frequency': congestion_periods.mean(),
            'monthly_congestion': monthly_congestion.to_dict(),
            'weekly_congestion': weekly_congestion.to_dict(),
            'congestion_events': len(congestion_events),
            'avg_congestion_duration': avg_congestion_duration,
            'max_congestion_duration': max([event['duration'] for event in congestion_events]) if congestion_events else 0
        }

    def _identify_congestion_events(self, congestion_series: pd.Series) -> List[Dict[str, Any]]:
        """Identify individual congestion events."""
        events = []

        in_event = False
        start_date = None

        for date, is_congested in congestion_series.items():
            if is_congested and not in_event:
                # Start of congestion event
                in_event = True
                start_date = date
            elif not is_congested and in_event:
                # End of congestion event
                in_event = False
                duration = (date - start_date).days
                events.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration': duration
                })

        # Handle ongoing event
        if in_event:
            duration = (congestion_series.index[-1] - start_date).days
            events.append({
                'start_date': start_date,
                'end_date': congestion_series.index[-1],
                'duration': duration
            })

        return events

    def model_weather_basis_impact(
        self,
        basis_series: pd.Series,
        temperature_data: pd.Series,
        heating_degree_days: Optional[pd.Series] = None,
        cooling_degree_days: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Model the impact of weather on regional basis.

        Args:
            basis_series: Historical basis data
            temperature_data: Temperature data for the region
            heating_degree_days: HDD series
            cooling_degree_days: CDD series

        Returns:
            Weather impact analysis
        """
        # Align data
        data = pd.DataFrame({
            'basis': basis_series,
            'temperature': temperature_data
        }).dropna()

        if heating_degree_days is not None:
            data['hdd'] = heating_degree_days
        if cooling_degree_days is not None:
            data['cdd'] = cooling_degree_days

        data = data.dropna()

        if len(data) < 30:
            return {'error': 'Insufficient data for weather impact analysis'}

        # Correlation analysis
        correlations = {}
        for col in ['temperature', 'hdd', 'cdd']:
            if col in data.columns:
                corr = np.corrcoef(data['basis'], data[col])[0, 1]
                correlations[col] = corr

        # Regression analysis
        X_cols = [col for col in ['temperature', 'hdd', 'cdd'] if col in data.columns]
        if X_cols:
            X = data[X_cols]
            y = data['basis']

            model = LinearRegression()
            model.fit(X, y)

            # Calculate R-squared
            r_squared = model.score(X, y)

            # Weather sensitivity coefficients
            coefficients = dict(zip(X_cols, model.coef_))

            return {
                'correlations': correlations,
                'coefficients': coefficients,
                'r_squared': r_squared,
                'intercept': model.intercept_,
                'weather_sensitivity': coefficients
            }
        else:
            return {'correlations': correlations}

    def optimize_regional_arbitrage(
        self,
        current_basis: Dict[str, float],
        transport_costs: Dict[str, float],
        pipeline_capacities: Dict[str, float],
        storage_costs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Optimize arbitrage opportunities across regions.

        Args:
            current_basis: Current basis levels by region
            transport_costs: Transport costs between regions
            pipeline_capacities: Available pipeline capacity
            storage_costs: Storage costs by region

        Returns:
            Arbitrage optimization results
        """
        regions = list(current_basis.keys())

        if len(regions) < 2:
            return {'strategy': 'no_opportunity', 'profit': 0}

        # Find arbitrage opportunities
        opportunities = []

        for i, region1 in enumerate(regions[:-1]):
            for region2 in regions[i+1:]:
                basis_diff = current_basis[region2] - current_basis[region1]
                transport_cost = transport_costs.get(f'{region1}_to_{region2}', 0)

                # Net profit per MMBtu
                net_profit = basis_diff - transport_cost

                if net_profit > 0:
                    capacity = pipeline_capacities.get(f'{region1}_to_{region2}', 0)
                    storage_cost = storage_costs.get(region2, 0) - storage_costs.get(region1, 0)

                    # Adjust for storage cost differential
                    adjusted_profit = net_profit - storage_cost

                    if adjusted_profit > 0:
                        opportunities.append({
                            'from_region': region1,
                            'to_region': region2,
                            'basis_differential': basis_diff,
                            'transport_cost': transport_cost,
                            'storage_cost_impact': storage_cost,
                            'net_profit_per_mmbtu': adjusted_profit,
                            'available_capacity': capacity,
                            'potential_daily_profit': adjusted_profit * capacity * 1000  # Convert to daily
                        })

        if not opportunities:
            return {'strategy': 'no_opportunity', 'profit': 0}

        # Select best opportunity
        best_opportunity = max(opportunities, key=lambda x: x['net_profit_per_mmbtu'])

        return {
            'strategy': 'regional_arbitrage',
            'best_opportunity': best_opportunity,
            'all_opportunities': opportunities,
            'total_potential_profit': sum(opp['potential_daily_profit'] for opp in opportunities),
            'regions_analyzed': len(regions)
        }

    def forecast_seasonal_basis_patterns(
        self,
        historical_basis: Dict[str, pd.Series],
        years_back: int = 5
    ) -> Dict[str, pd.Series]:
        """
        Forecast seasonal basis patterns based on historical data.

        Args:
            historical_basis: Historical basis data by region
            years_back: Years of historical data to analyze

        Returns:
            Seasonal basis forecasts
        """
        forecasts = {}

        for region, basis_series in historical_basis.items():
            # Calculate monthly average basis
            monthly_basis = basis_series.groupby(basis_series.index.month).mean()

            # Calculate seasonal volatility
            monthly_volatility = basis_series.groupby(basis_series.index.month).std()

            # Simple seasonal forecast (use historical averages)
            forecast_months = 12
            seasonal_forecast = pd.Series(index=range(1, forecast_months + 1))

            for month in range(1, forecast_months + 1):
                if month in monthly_basis.index:
                    seasonal_forecast[month] = monthly_basis[month]
                else:
                    # Use overall average if no historical data
                    seasonal_forecast[month] = basis_series.mean()

            forecasts[region] = seasonal_forecast

        return forecasts

    def calculate_basis_risk_metrics(
        self,
        basis_series: pd.Series,
        benchmark_return: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate risk metrics for basis positions.

        Args:
            basis_series: Historical basis data
            benchmark_return: Benchmark return series for correlation

        Returns:
            Basis risk metrics
        """
        if len(basis_series) < 30:
            return {'error': 'Insufficient data for risk metrics'}

        # Basic statistics
        mean_basis = basis_series.mean()
        std_basis = basis_series.std()
        min_basis = basis_series.min()
        max_basis = basis_series.max()

        # Value at Risk (95% confidence)
        var_95 = basis_series.quantile(0.05)

        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = basis_series[basis_series <= var_95].mean()

        # Maximum drawdown
        cumulative = basis_series.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = mean_basis / std_basis if std_basis > 0 else 0

        # Correlation with benchmark (if provided)
        correlation = 0
        if benchmark_return is not None:
            aligned_data = pd.DataFrame({
                'basis': basis_series,
                'benchmark': benchmark_return
            }).dropna()

            if len(aligned_data) > 10:
                correlation = np.corrcoef(aligned_data['basis'], aligned_data['benchmark'])[0, 1]

        return {
            'mean_basis': mean_basis,
            'basis_volatility': std_basis,
            'basis_range': max_basis - min_basis,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'correlation_with_benchmark': correlation,
            'data_points': len(basis_series)
        }

    def analyze_storage_basis_impact(
        self,
        basis_series: pd.Series,
        storage_levels: pd.Series,
        injection_withdrawal_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze how storage levels impact regional basis.

        Args:
            basis_series: Historical basis data
            storage_levels: Storage level data
            injection_withdrawal_data: Injection/withdrawal rates

        Returns:
            Storage impact analysis
        """
        # Align data
        data = pd.DataFrame({
            'basis': basis_series,
            'storage_level': storage_levels
        }).dropna()

        if injection_withdrawal_data is not None:
            data['net_flow'] = injection_withdrawal_data

        data = data.dropna()

        if len(data) < 30:
            return {'error': 'Insufficient data for storage analysis'}

        # Correlation between storage levels and basis
        storage_correlation = np.corrcoef(data['basis'], data['storage_level'])[0, 1]

        # Regression analysis
        X = data[['storage_level']]
        y = data['basis']

        model = LinearRegression()
        model.fit(X, y)

        storage_sensitivity = model.coef_[0]

        # Analyze storage level regimes
        storage_percentiles = data['storage_level'].quantile([0.25, 0.5, 0.75])

        low_storage_basis = data[data['storage_level'] <= storage_percentiles[0.25]]['basis'].mean()
        high_storage_basis = data[data['storage_level'] >= storage_percentiles[0.75]]['basis'].mean()

        storage_regime_impact = high_storage_basis - low_storage_basis

        return {
            'storage_correlation': storage_correlation,
            'storage_sensitivity': storage_sensitivity,
            'low_storage_basis': low_storage_basis,
            'high_storage_basis': high_storage_basis,
            'storage_regime_impact': storage_regime_impact,
            'storage_percentiles': storage_percentiles.to_dict(),
            'r_squared': model.score(X, y)
        }
