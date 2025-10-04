"""
Seasonal Demand Forecasting Model

Advanced forecasting model for seasonal natural gas demand patterns:
- Weather-driven demand modeling
- Economic activity impact analysis
- Calendar effect adjustments
- Peak demand prediction
- Demand elasticity estimation
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SeasonalDemandForecast:
    """
    Seasonal demand forecasting and analysis model.

    Features:
    - Weather-adjusted demand forecasting
    - Economic indicator integration
    - Calendar effect modeling
    - Peak demand prediction
    - Demand elasticity analysis
    """

    def __init__(self):
        # Base demand levels by region (MMcf/d)
        self.regional_base_demand = {
            'northeast': 15000,
            'midwest': 12000,
            'south': 20000,
            'west': 8000,
            'california': 6000
        }

        # Weather sensitivity coefficients
        self.weather_sensitivity = {
            'heating': 0.15,  # 15% increase per HDD
            'cooling': 0.08   # 8% increase per CDD
        }

    def forecast_seasonal_demand(
        self,
        historical_demand: pd.Series,
        temperature_forecast: pd.Series,
        economic_indicators: Optional[Dict[str, pd.Series]] = None,
        region: str = 'northeast',
        forecast_horizon: int = 90
    ) -> pd.Series:
        """
        Forecast seasonal demand with weather and economic adjustments.

        Args:
            historical_demand: Historical demand data
            temperature_forecast: Temperature forecast
            economic_indicators: Economic indicators (GDP, industrial production)
            region: Region for base demand and sensitivity
            forecast_horizon: Days to forecast

        Returns:
            Demand forecast series
        """
        logger.info(f"Forecasting seasonal demand for {region}")

        # Get base demand for region
        base_demand = self.regional_base_demand.get(region, 10000)

        # Calculate degree days for forecast period
        degree_days_forecast = self._calculate_degree_days_from_temperature(temperature_forecast)

        # Build demand model using historical data
        demand_model = self._build_demand_model(historical_demand, region)

        # Generate base forecast (trend + seasonal)
        base_forecast = self._generate_base_forecast(
            historical_demand, forecast_horizon, demand_model
        )

        # Apply weather adjustments
        weather_adjusted_forecast = self._apply_weather_adjustments(
            base_forecast, degree_days_forecast, region
        )

        # Apply economic adjustments if provided
        if economic_indicators:
            final_forecast = self._apply_economic_adjustments(
                weather_adjusted_forecast, economic_indicators, region
            )
        else:
            final_forecast = weather_adjusted_forecast

        return final_forecast

    def _calculate_degree_days_from_temperature(self, temperatures: pd.Series) -> pd.DataFrame:
        """Calculate heating and cooling degree days from temperature data."""
        degree_days = pd.DataFrame(index=temperatures.index)

        for date, temp in temperatures.items():
            if pd.isna(temp):
                degree_days.loc[date, 'heating_degree_days'] = np.nan
                degree_days.loc[date, 'cooling_degree_days'] = np.nan
            else:
                # Heating degree days (base 65°F)
                hdd = max(0, 65 - temp)
                degree_days.loc[date, 'heating_degree_days'] = hdd

                # Cooling degree days (base 65°F)
                cdd = max(0, temp - 65)
                degree_days.loc[date, 'cooling_degree_days'] = cdd

        return degree_days

    def _build_demand_model(self, historical_demand: pd.Series, region: str) -> Dict[str, Any]:
        """Build demand forecasting model from historical data."""
        # Prepare features from historical data
        features = pd.DataFrame(index=historical_demand.index)

        # Time-based features
        features['month'] = features.index.month
        features['day_of_year'] = features.index.dayofyear
        features['day_of_week'] = features.index.dayofweek
        features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)

        # Seasonal features
        features['is_heating_season'] = features.index.month.isin([12, 1, 2, 3]).astype(int)
        features['is_cooling_season'] = features.index.month.isin([6, 7, 8]).astype(int)

        # Lag features
        for lag in [1, 7, 30]:
            features[f'demand_lag_{lag}'] = historical_demand.shift(lag)

        # Rolling statistics
        for window in [7, 30]:
            features[f'demand_mean_{window}'] = historical_demand.rolling(window).mean()
            features[f'demand_std_{window}'] = historical_demand.rolling(window).std()

        # Remove NaN values
        valid_data = features.dropna()
        valid_demand = historical_demand[valid_data.index]

        if len(valid_data) < 50:
            # Fallback to simple model
            return {
                'model_type': 'simple',
                'base_demand': historical_demand.mean(),
                'trend': historical_demand.diff().mean(),
                'seasonal_pattern': historical_demand.groupby(historical_demand.index.month).mean()
            }

        # Fit linear regression model
        X = valid_data
        y = valid_demand

        model = LinearRegression()
        model.fit(X, y)

        # Calculate model performance
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        return {
            'model_type': 'linear_regression',
            'model': model,
            'features': list(valid_data.columns),
            'train_mae': mae,
            'train_rmse': rmse,
            'base_demand': historical_demand.mean(),
            'feature_coefficients': dict(zip(valid_data.columns, model.coef_))
        }

    def _generate_base_forecast(
        self,
        historical_demand: pd.Series,
        horizon: int,
        model: Dict[str, Any]
    ) -> pd.Series:
        """Generate base demand forecast (trend + seasonal)."""
        last_date = historical_demand.index[-1]
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

        if model['model_type'] == 'simple':
            # Simple forecast using mean and trend
            base_demand = model['base_demand']
            trend = model['trend']

            forecast_values = []
            for i in range(horizon):
                seasonal_month = (last_date.month + i) % 12 + 1
                seasonal_factor = model['seasonal_pattern'].get(seasonal_month, 1.0)
                forecast_value = base_demand + trend * (i + 1) * seasonal_factor
                forecast_values.append(forecast_value)

            return pd.Series(forecast_values, index=forecast_dates)

        else:
            # Use regression model
            forecast_features = self._create_forecast_features(forecast_dates, model)
            forecast_values = model['model'].predict(forecast_features)

            return pd.Series(forecast_values, index=forecast_dates)

    def _create_forecast_features(self, forecast_dates: pd.DatetimeIndex, model: Dict[str, Any]) -> pd.DataFrame:
        """Create features for forecast period."""
        features = pd.DataFrame(index=forecast_dates)

        # Time-based features
        features['month'] = features.index.month
        features['day_of_year'] = features.index.dayofyear
        features['day_of_week'] = features.index.dayofweek
        features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)

        # Seasonal features
        features['is_heating_season'] = features.index.month.isin([12, 1, 2, 3]).astype(int)
        features['is_cooling_season'] = features.index.month.isin([6, 7, 8]).astype(int)

        # Use recent historical data for lag features
        recent_demand = model.get('recent_demand', pd.Series([model['base_demand']] * 30))

        for lag in [1, 7, 30]:
            if lag <= len(recent_demand):
                features[f'demand_lag_{lag}'] = recent_demand.iloc[-lag]
            else:
                features[f'demand_lag_{lag}'] = model['base_demand']

        # Rolling statistics (use recent data)
        for window in [7, 30]:
            if window <= len(recent_demand):
                features[f'demand_mean_{window}'] = recent_demand.iloc[-window:].mean()
                features[f'demand_std_{window}'] = recent_demand.iloc[-window:].std()
            else:
                features[f'demand_mean_{window}'] = model['base_demand']
                features[f'demand_std_{window}'] = recent_demand.std()

        return features[model['features']]  # Return only features used in model

    def _apply_weather_adjustments(
        self,
        base_forecast: pd.Series,
        degree_days_forecast: pd.DataFrame,
        region: str
    ) -> pd.Series:
        """Apply weather adjustments to base forecast."""
        adjusted_forecast = base_forecast.copy()

        for date, base_demand in base_forecast.items():
            if date in degree_days_forecast.index:
                hdd = degree_days_forecast.loc[date, 'heating_degree_days']
                cdd = degree_days_forecast.loc[date, 'cooling_degree_days']

                # Apply weather sensitivity
                heating_adjustment = 1 + (hdd / 100) * self.weather_sensitivity['heating']
                cooling_adjustment = 1 + (cdd / 100) * self.weather_sensitivity['cooling']

                # Combine adjustments (heating and cooling can overlap)
                weather_adjustment = heating_adjustment * cooling_adjustment

                adjusted_forecast[date] = base_demand * weather_adjustment

        return adjusted_forecast

    def _apply_economic_adjustments(
        self,
        weather_forecast: pd.Series,
        economic_indicators: Dict[str, pd.Series],
        region: str
    ) -> pd.Series:
        """Apply economic indicator adjustments to forecast."""
        adjusted_forecast = weather_forecast.copy()

        # Calculate economic growth impact
        # Simple model: 1% GDP growth = 0.5% demand increase
        gdp_impact = 0

        if 'gdp_growth' in economic_indicators:
            gdp_series = economic_indicators['gdp_growth']
            if len(gdp_series) > 0:
                avg_gdp_growth = gdp_series.mean()
                gdp_impact = avg_gdp_growth * 0.5  # 0.5% demand per 1% GDP growth

        # Industrial production impact
        industrial_impact = 0

        if 'industrial_production' in economic_indicators:
            ip_series = economic_indicators['industrial_production']
            if len(ip_series) > 0:
                avg_ip_growth = ip_series.pct_change().mean()
                industrial_impact = avg_ip_growth * 0.3  # 0.3% demand per 1% IP growth

        # Apply economic adjustments
        total_economic_impact = gdp_impact + industrial_impact

        for date in adjusted_forecast.index:
            adjusted_forecast[date] *= (1 + total_economic_impact)

        return adjusted_forecast

    def predict_peak_demand(
        self,
        historical_demand: pd.Series,
        extreme_weather_scenarios: List[Dict[str, Any]],
        return_period: int = 10  # years
    ) -> Dict[str, float]:
        """
        Predict peak demand under extreme weather conditions.

        Args:
            historical_demand: Historical demand data
            extreme_weather_scenarios: Weather scenarios for peak analysis
            return_period: Return period for extreme events (years)

        Returns:
            Peak demand predictions
        """
        # Calculate historical peak demand
        historical_peak = historical_demand.max()
        historical_99th_percentile = historical_demand.quantile(0.99)

        # Analyze extreme weather impact
        extreme_demand_multipliers = []

        for scenario in extreme_weather_scenarios:
            # Calculate degree days for extreme scenario
            temp = scenario['temperature']
            duration = scenario['duration_days']

            extreme_hdd = max(0, 65 - temp) * duration
            extreme_cdd = max(0, temp - 65) * duration

            # Calculate demand multiplier for extreme weather
            heating_multiplier = 1 + (extreme_hdd / 100) * self.weather_sensitivity['heating']
            cooling_multiplier = 1 + (extreme_cdd / 100) * self.weather_sensitivity['cooling']

            extreme_multiplier = heating_multiplier * cooling_multiplier
            extreme_demand_multipliers.append(extreme_multiplier)

        # Calculate peak demand under extreme conditions
        avg_extreme_multiplier = np.mean(extreme_demand_multipliers)
        extreme_peak_demand = historical_99th_percentile * avg_extreme_multiplier

        # Statistical peak demand (using extreme value theory)
        # Simplified: assume normal distribution for peaks
        peak_std = historical_demand.std()
        statistical_peak = historical_demand.mean() + 3 * peak_std  # 3-sigma event

        # Return period adjustment (simplified)
        return_period_multiplier = 1 + 0.1 * np.log(return_period)

        return_period_peak = extreme_peak_demand * return_period_multiplier

        return {
            'historical_peak': historical_peak,
            '99th_percentile_demand': historical_99th_percentile,
            'extreme_weather_peak': extreme_peak_demand,
            'statistical_peak': statistical_peak,
            'return_period_peak': return_period_peak,
            'avg_extreme_multiplier': avg_extreme_multiplier,
            'return_period_years': return_period,
            'extreme_scenarios_analyzed': len(extreme_weather_scenarios)
        }

    def estimate_demand_elasticity(
        self,
        demand_data: pd.Series,
        price_data: pd.Series,
        temperature_adjusted: bool = True,
        method: str = 'regression'
    ) -> Dict[str, float]:
        """
        Estimate price elasticity of demand for natural gas.

        Args:
            demand_data: Demand/consumption data
            price_data: Price data
            temperature_adjusted: Whether to adjust for temperature
            method: 'regression' or 'correlation'

        Returns:
            Demand elasticity estimates
        """
        # Align data
        data = pd.DataFrame({
            'demand': demand_data,
            'price': price_data
        }).dropna()

        if len(data) < 30:
            return {'error': 'Insufficient data for elasticity estimation'}

        if method == 'correlation':
            # Simple correlation-based elasticity
            correlation = np.corrcoef(data['demand'], data['price'])[0, 1]

            # Convert correlation to elasticity approximation
            demand_std = data['demand'].std()
            price_std = data['price'].std()
            price_mean = data['price'].mean()

            elasticity = correlation * (demand_std / price_std) * (price_mean / data['demand'].mean())

        elif method == 'regression':
            # Regression-based elasticity
            X = data[['price']]
            y = data['demand']

            model = LinearRegression()
            model.fit(X, y)

            # Elasticity = coefficient * (mean price / mean demand)
            elasticity = model.coef_[0] * (data['price'].mean() / data['demand'].mean())

            # Calculate R-squared
            r_squared = model.score(X, y)

            return {
                'elasticity': elasticity,
                'r_squared': r_squared,
                'coefficient': model.coef_[0],
                'intercept': model.intercept_,
                'method': 'regression',
                'temperature_adjusted': temperature_adjusted,
                'data_points': len(data)
            }

        else:
            raise ValueError(f"Unknown elasticity method: {method}")

        return {
            'elasticity': elasticity,
            'method': method,
            'temperature_adjusted': temperature_adjusted,
            'data_points': len(data)
        }

    def analyze_demand_response(
        self,
        demand_data: pd.Series,
        price_changes: pd.Series,
        response_lag: int = 7  # days
    ) -> Dict[str, Any]:
        """
        Analyze demand response to price changes.

        Args:
            demand_data: Demand data
            price_changes: Price change data
            response_lag: Lag for demand response (days)

        Returns:
            Demand response analysis
        """
        # Calculate lagged correlations
        max_lag = min(response_lag * 2, len(demand_data) // 10)  # Reasonable max lag

        correlations = []
        for lag in range(1, max_lag + 1):
            lagged_prices = price_changes.shift(lag)
            correlation = np.corrcoef(demand_data, lagged_prices)[0, 1]
            correlations.append({
                'lag_days': lag,
                'correlation': correlation if not np.isnan(correlation) else 0
            })

        # Find optimal lag (highest absolute correlation)
        best_lag = max(correlations, key=lambda x: abs(x['correlation']))

        # Analyze response magnitude
        if abs(best_lag['correlation']) > 0.1:  # Significant correlation
            # Calculate response coefficient
            aligned_data = pd.DataFrame({
                'demand': demand_data,
                'price_change': price_changes.shift(best_lag['lag_days'])
            }).dropna()

            if len(aligned_data) > 20:
                response_model = LinearRegression()
                response_model.fit(aligned_data[['price_change']], aligned_data['demand'])

                response_coefficient = response_model.coef_[0]

                return {
                    'optimal_lag': best_lag['lag_days'],
                    'correlation': best_lag['correlation'],
                    'response_coefficient': response_coefficient,
                    'response_magnitude': abs(response_coefficient),
                    'response_direction': 'negative' if response_coefficient < 0 else 'positive',
                    'significant_response': True,
                    'data_points': len(aligned_data)
                }

        return {
            'optimal_lag': best_lag['lag_days'],
            'correlation': best_lag['correlation'],
            'response_coefficient': 0,
            'response_magnitude': 0,
            'response_direction': 'none',
            'significant_response': False,
            'data_points': len(demand_data)
        }

    def forecast_demand_volatility(
        self,
        historical_demand: pd.Series,
        forecast_horizon: int = 30,
        volatility_model: str = 'garch'
    ) -> pd.Series:
        """
        Forecast demand volatility for risk management.

        Args:
            historical_demand: Historical demand data
            forecast_horizon: Days to forecast volatility
            volatility_model: 'garch', 'rolling_std', or 'ewma'

        Returns:
            Volatility forecast series
        """
        if volatility_model == 'rolling_std':
            # Rolling standard deviation
            window = min(30, len(historical_demand) // 4)
            volatility = historical_demand.rolling(window).std()

            # Forecast: use recent volatility levels
            recent_volatility = volatility.iloc[-window:].mean()
            volatility_forecast = pd.Series([recent_volatility] * forecast_horizon,
                                          index=pd.date_range(
                                              historical_demand.index[-1] + pd.Timedelta(days=1),
                                              periods=forecast_horizon
                                          ))

        elif volatility_model == 'ewma':
            # Exponentially weighted moving average volatility
            alpha = 0.94  # EWMA parameter
            returns = historical_demand.pct_change().dropna()

            # Calculate EWMA variance
            variance = returns.ewm(alpha=alpha).var()

            # Convert to standard deviation
            volatility = np.sqrt(variance)

            # Forecast: use recent volatility trend
            recent_volatility = volatility.iloc[-30:].mean()
            volatility_forecast = pd.Series([recent_volatility] * forecast_horizon,
                                          index=pd.date_range(
                                              historical_demand.index[-1] + pd.Timedelta(days=1),
                                              periods=forecast_horizon
                                          ))

        else:
            # Default to rolling standard deviation
            window = min(30, len(historical_demand) // 4)
            volatility = historical_demand.rolling(window).std()

            recent_volatility = volatility.iloc[-window:].mean()
            volatility_forecast = pd.Series([recent_volatility] * forecast_horizon,
                                          index=pd.date_range(
                                              historical_demand.index[-1] + pd.Timedelta(days=1),
                                              periods=forecast_horizon
                                          ))

        return volatility_forecast

    def generate_demand_scenarios(
        self,
        base_forecast: pd.Series,
        scenario_parameters: Dict[str, Dict[str, float]]
    ) -> Dict[str, pd.Series]:
        """
        Generate demand scenarios for risk analysis.

        Args:
            base_forecast: Base demand forecast
            scenario_parameters: Scenario definitions with multipliers

        Returns:
            Dictionary of scenario forecasts
        """
        scenarios = {}

        for scenario_name, params in scenario_parameters.items():
            scenario_forecast = base_forecast.copy()

            # Apply scenario multipliers
            weather_multiplier = params.get('weather_multiplier', 1.0)
            economic_multiplier = params.get('economic_multiplier', 1.0)
            random_shock = params.get('random_shock', 0.0)

            # Apply multipliers
            scenario_forecast = scenario_forecast * weather_multiplier * economic_multiplier

            # Add random shock if specified
            if random_shock > 0:
                noise = np.random.normal(0, random_shock, len(scenario_forecast))
                scenario_forecast = scenario_forecast * (1 + noise)

            scenarios[scenario_name] = scenario_forecast

        return scenarios
