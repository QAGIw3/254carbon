"""
Weather-Demand Impact Models

Models for analyzing weather impact on energy demand and prices:
- Heating and cooling degree day calculations
- Weather sensitivity analysis
- Temperature-price elasticity modeling
- Extreme weather event impact assessment
- Seasonal demand forecasting
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class WeatherDemandModel:
    """
    Weather impact on energy demand and pricing model.

    Features:
    - Heating/cooling degree day calculations
    - Temperature sensitivity analysis
    - Price elasticity modeling
    - Extreme weather impact assessment
    """

    def __init__(self):
        self.baseline_temperatures = {
            'heating_base': 65,  # °F - temperature below which heating is needed
            'cooling_base': 65,  # °F - temperature above which cooling is needed
        }

        self.regional_baselines = {
            'northeast': {'heating': 60, 'cooling': 70},
            'midwest': {'heating': 65, 'cooling': 72},
            'south': {'heating': 68, 'cooling': 75},
            'west': {'heating': 62, 'cooling': 70},
            'california': {'heating': 65, 'cooling': 75}
        }

    def calculate_heating_degree_days(
        self,
        temperatures: pd.Series,
        base_temp: Optional[float] = None,
        region: Optional[str] = None
    ) -> pd.Series:
        """
        Calculate heating degree days (HDD).

        HDD = max(0, base_temperature - average_daily_temperature)
        """
        if base_temp is None:
            if region and region in self.regional_baselines:
                base_temp = self.regional_baselines[region]['heating']
            else:
                base_temp = self.baseline_temperatures['heating_base']

        # Calculate HDD for each day
        hdd = pd.Series(index=temperatures.index)

        for date, temp in temperatures.items():
            if pd.isna(temp):
                hdd[date] = np.nan
            else:
                # Use average of daily high and low if available
                # For now, assume single daily temperature reading
                daily_hdd = max(0, base_temp - temp)
                hdd[date] = daily_hdd

        return hdd

    def calculate_cooling_degree_days(
        self,
        temperatures: pd.Series,
        base_temp: Optional[float] = None,
        region: Optional[str] = None
    ) -> pd.Series:
        """
        Calculate cooling degree days (CDD).

        CDD = max(0, average_daily_temperature - base_temperature)
        """
        if base_temp is None:
            if region and region in self.regional_baselines:
                base_temp = self.regional_baselines[region]['cooling']
            else:
                base_temp = self.baseline_temperatures['cooling_base']

        # Calculate CDD for each day
        cdd = pd.Series(index=temperatures.index)

        for date, temp in temperatures.items():
            if pd.isna(temp):
                cdd[date] = np.nan
            else:
                daily_cdd = max(0, temp - base_temp)
                cdd[date] = daily_cdd

        return cdd

    def calculate_degree_days(
        self,
        temperatures: pd.Series,
        region: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate both heating and cooling degree days.

        Returns:
            DataFrame with HDD and CDD columns
        """
        hdd = self.calculate_heating_degree_days(temperatures, region=region)
        cdd = self.calculate_cooling_degree_days(temperatures, region=region)

        return pd.DataFrame({
            'heating_degree_days': hdd,
            'cooling_degree_days': cdd,
            'total_degree_days': hdd + cdd
        })

    def analyze_temperature_sensitivity(
        self,
        prices: pd.Series,
        temperatures: pd.Series,
        degree_days: Optional[pd.DataFrame] = None,
        polynomial_degree: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze temperature sensitivity of energy prices.

        Args:
            prices: Energy price series
            temperatures: Temperature series
            degree_days: Pre-calculated degree days
            polynomial_degree: Degree of polynomial for temperature relationship

        Returns:
            Temperature sensitivity analysis results
        """
        logger.info("Analyzing temperature sensitivity of energy prices")

        # Align data
        data = pd.DataFrame({
            'price': prices,
            'temperature': temperatures
        }).dropna()

        if len(data) < 30:
            return {'error': 'Insufficient data for temperature sensitivity analysis'}

        # Calculate degree days if not provided
        if degree_days is None:
            degree_days = self.calculate_degree_days(temperatures)

        # Merge degree days with price data
        analysis_data = data.merge(
            degree_days,
            left_index=True,
            right_index=True,
            how='inner'
        )

        if len(analysis_data) < 30:
            return {'error': 'Insufficient overlapping data'}

        # Fit polynomial relationship between temperature and price
        X = analysis_data[['temperature', 'heating_degree_days', 'cooling_degree_days']]
        y = analysis_data['price']

        # Polynomial features for temperature
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
        X_poly = poly.fit_transform(X)

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_poly, y)

        # Calculate R-squared
        r_squared = model.score(X_poly, y)

        # Temperature sensitivity coefficients
        coefficients = model.coef_

        # Analyze heating vs cooling sensitivity
        heating_sensitivity = np.corrcoef(
            analysis_data['heating_degree_days'],
            analysis_data['price']
        )[0, 1]

        cooling_sensitivity = np.corrcoef(
            analysis_data['cooling_degree_days'],
            analysis_data['price']
        )[0, 1]

        # Find optimal temperature (minimum price)
        temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)
        price_predictions = []

        for temp in temp_range:
            # Create feature vector for prediction
            temp_hdd = max(0, 65 - temp)  # Simple HDD calculation
            temp_cdd = max(0, temp - 65)  # Simple CDD calculation

            X_pred = poly.transform([[temp, temp_hdd, temp_cdd]])
            pred_price = model.predict(X_pred)[0]
            price_predictions.append(pred_price)

        optimal_temp_idx = np.argmin(price_predictions)
        optimal_temperature = temp_range[optimal_temp_idx]

        return {
            'r_squared': r_squared,
            'coefficients': coefficients,
            'heating_sensitivity': heating_sensitivity,
            'cooling_sensitivity': cooling_sensitivity,
            'optimal_temperature': optimal_temperature,
            'temperature_range': temp_range,
            'predicted_prices': price_predictions,
            'polynomial_degree': polynomial_degree
        }

    def forecast_demand_from_weather(
        self,
        temperature_forecast: pd.Series,
        historical_prices: pd.Series,
        historical_temperatures: pd.Series,
        base_demand: float = 1000,  # Base demand in MW or MMBtu
        region: Optional[str] = None
    ) -> pd.Series:
        """
        Forecast energy demand based on weather forecasts.

        Args:
            temperature_forecast: Temperature forecast series
            historical_prices: Historical price data for model calibration
            historical_temperatures: Historical temperature data
            base_demand: Base demand level
            region: Region for regional baselines

        Returns:
            Demand forecast series
        """
        # Calculate degree days for forecast
        degree_days_forecast = self.calculate_degree_days(temperature_forecast, region)

        # Analyze historical relationship between degree days and prices
        historical_degree_days = self.calculate_degree_days(historical_temperatures, region)

        # Merge historical data
        historical_data = pd.DataFrame({
            'price': historical_prices,
            'hdd': historical_degree_days['heating_degree_days'],
            'cdd': historical_degree_days['cooling_degree_days']
        }).dropna()

        if len(historical_data) < 30:
            logger.warning("Insufficient historical data for demand forecasting")
            # Simple fallback: linear relationship
            demand_forecast = base_demand * (1 + 0.01 * (degree_days_forecast['total_degree_days'] / 100))
            return demand_forecast

        # Fit demand model
        X = historical_data[['hdd', 'cdd']]
        y = historical_data['price']

        demand_model = LinearRegression()
        demand_model.fit(X, y)

        # Predict demand from weather forecast
        demand_forecast = demand_model.predict(degree_days_forecast[['heating_degree_days', 'cooling_degree_days']])

        # Scale to base demand level
        demand_forecast = base_demand * (1 + (demand_forecast - demand_forecast.mean()) / demand_forecast.std())

        return pd.Series(demand_forecast, index=temperature_forecast.index)

    def detect_extreme_weather_impact(
        self,
        prices: pd.Series,
        temperatures: pd.Series,
        threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect and quantify impact of extreme weather events on prices.

        Args:
            prices: Price series
            temperatures: Temperature series
            threshold_std: Standard deviations for extreme event detection

        Returns:
            Analysis of extreme weather impacts
        """
        # Calculate temperature anomalies
        temp_mean = temperatures.mean()
        temp_std = temperatures.std()

        extreme_cold = temperatures < (temp_mean - threshold_std * temp_std)
        extreme_hot = temperatures > (temp_mean + threshold_std * temp_std)

        # Analyze price impact during extreme events
        cold_impact = self._analyze_weather_event_impact(
            prices[extreme_cold], "extreme_cold"
        )

        hot_impact = self._analyze_weather_event_impact(
            prices[extreme_hot], "extreme_hot"
        )

        # Overall extreme weather impact
        extreme_events = extreme_cold | extreme_hot
        normal_events = ~extreme_events

        if len(prices[extreme_events]) > 0 and len(prices[normal_events]) > 0:
            extreme_avg_price = prices[extreme_events].mean()
            normal_avg_price = prices[normal_events].mean()
            price_premium = ((extreme_avg_price / normal_avg_price) - 1) * 100
        else:
            price_premium = 0

        return {
            'extreme_cold_events': extreme_cold.sum(),
            'extreme_hot_events': extreme_hot.sum(),
            'total_extreme_events': extreme_events.sum(),
            'cold_impact': cold_impact,
            'hot_impact': hot_impact,
            'price_premium_during_extremes': price_premium,
            'temperature_thresholds': {
                'extreme_cold': temp_mean - threshold_std * temp_std,
                'extreme_hot': temp_mean + threshold_std * temp_std
            }
        }

    def _analyze_weather_event_impact(self, prices: pd.Series, event_type: str) -> Dict[str, float]:
        """Analyze price impact during specific weather events."""
        if len(prices) == 0:
            return {
                'avg_price': 0,
                'max_price': 0,
                'price_volatility': 0,
                'duration': 0
            }

        return {
            'avg_price': prices.mean(),
            'max_price': prices.max(),
            'price_volatility': prices.std(),
            'duration': len(prices)
        }

    def calculate_seasonal_demand_patterns(
        self,
        prices: pd.Series,
        temperatures: pd.Series,
        years_back: int = 5
    ) -> Dict[str, pd.Series]:
        """
        Calculate seasonal demand patterns based on historical data.

        Args:
            prices: Historical price data
            temperatures: Historical temperature data
            years_back: Number of years of historical data to analyze

        Returns:
            Seasonal patterns by month and day of week
        """
        # Create date range for analysis
        end_date = prices.index.max()
        start_date = end_date - pd.DateOffset(years=years_back)

        # Filter data to analysis period
        mask = (prices.index >= start_date) & (prices.index <= end_date)
        analysis_prices = prices[mask]
        analysis_temps = temperatures[mask]

        if len(analysis_prices) == 0:
            return {'error': 'No data in analysis period'}

        # Monthly patterns
        monthly_prices = analysis_prices.groupby(analysis_prices.index.month).agg(['mean', 'std', 'count'])
        monthly_temps = analysis_temps.groupby(analysis_temps.index.month).agg(['mean', 'std'])

        # Weekly patterns
        weekly_prices = analysis_prices.groupby(analysis_prices.index.weekday).agg(['mean', 'std', 'count'])
        weekly_temps = analysis_temps.groupby(analysis_temps.index.weekday).agg(['mean', 'std'])

        # Calculate heating/cooling sensitivity by month
        monthly_sensitivity = {}
        for month in range(1, 13):
            month_mask = analysis_prices.index.month == month
            if month_mask.sum() > 10:
                month_prices = analysis_prices[month_mask]
                month_temps = analysis_temps[month_mask]

                if len(month_prices) > 0 and len(month_temps) > 0:
                    correlation = np.corrcoef(month_prices, month_temps)[0, 1]
                    monthly_sensitivity[month] = correlation

        return {
            'monthly_price_patterns': monthly_prices['mean'],
            'monthly_temp_patterns': monthly_temps['mean'],
            'weekly_price_patterns': weekly_prices['mean'],
            'weekly_temp_patterns': weekly_temps['mean'],
            'monthly_sensitivity': monthly_sensitivity,
            'analysis_period': f'{start_date.date()} to {end_date.date()}'
        }

    def estimate_price_elasticity(
        self,
        prices: pd.Series,
        demand_proxy: pd.Series,
        temperature_adjusted: bool = True
    ) -> Dict[str, float]:
        """
        Estimate price elasticity of demand for energy.

        Args:
            prices: Price series
            demand_proxy: Proxy for demand (can be volume, consumption, etc.)
            temperature_adjusted: Whether to adjust for temperature effects

        Returns:
            Price elasticity estimates
        """
        # Align data
        data = pd.DataFrame({
            'price': prices,
            'demand': demand_proxy
        }).dropna()

        if len(data) < 20:
            return {'error': 'Insufficient data for elasticity estimation'}

        # Simple elasticity calculation using correlation
        price_demand_corr = np.corrcoef(data['price'], data['demand'])[0, 1]

        # Elasticity = % change in demand / % change in price
        # Approximation using correlation and standard deviations
        price_std = data['price'].std()
        demand_std = data['demand'].std()
        price_mean = data['price'].mean()

        if price_std > 0 and price_mean > 0:
            elasticity = (demand_std / demand_std) / (price_std / price_mean) * price_demand_corr
        else:
            elasticity = 0

        # More sophisticated elasticity using regression
        X = data[['price']]
        y = data['demand']

        elasticity_model = LinearRegression()
        elasticity_model.fit(X, y)

        # Elasticity coefficient from regression
        regression_elasticity = elasticity_model.coef_[0] * (price_mean / demand_std)

        return {
            'correlation_based_elasticity': elasticity,
            'regression_based_elasticity': regression_elasticity,
            'correlation_coefficient': price_demand_corr,
            'price_volatility': price_std,
            'demand_volatility': demand_std,
            'temperature_adjusted': temperature_adjusted
        }
