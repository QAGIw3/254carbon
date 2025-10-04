"""
RIN Price Forecasting Model

Advanced model for forecasting Renewable Identification Numbers (RIN) prices:
- D4, D5, D6 RIN price forecasting
- Compliance demand modeling
- Supply constraint analysis
- Policy impact assessment
- Market sentiment analysis
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class RINPriceForecast:
    """
    RIN price forecasting and analysis model.

    Features:
    - Multi-RIN category forecasting
    - Compliance demand modeling
    - Supply constraint analysis
    - Policy scenario modeling
    """

    def __init__(self):
        # RIN market fundamentals
        self.rin_categories = {
            'D4': {
                'description': 'Biomass-based diesel RIN',
                'ethanol_equivalent': 1.5,
                'typical_range': (0.80, 1.50),
                'volatility': 0.25
            },
            'D5': {
                'description': 'Advanced biofuels RIN',
                'ethanol_equivalent': 1.0,
                'typical_range': (0.60, 1.20),
                'volatility': 0.30
            },
            'D6': {
                'description': 'Renewable fuels RIN',
                'ethanol_equivalent': 1.0,
                'typical_range': (0.30, 0.80),
                'volatility': 0.35
            }
        }

        # Compliance requirements (simplified)
        self.compliance_requirements = {
            'total_renewable_volume': 15.0,  # Billion gallons
            'advanced_biofuels': 0.75,       # Billion gallons
            'biomass_diesel': 1.0,           # Billion gallons
            'cellulosic': 0.25               # Billion gallons
        }

    def forecast_rin_prices(
        self,
        historical_prices: Dict[str, pd.Series],
        compliance_forecast: Optional[pd.Series] = None,
        supply_forecast: Optional[Dict[str, pd.Series]] = None,
        forecast_horizon: int = 90
    ) -> Dict[str, pd.Series]:
        """
        Forecast RIN prices for different categories.

        Args:
            historical_prices: Historical RIN price data by category
            compliance_forecast: Forecasted compliance requirements
            supply_forecast: Forecasted RIN supply by category
            forecast_horizon: Days to forecast

        Returns:
            RIN price forecasts by category
        """
        forecasts = {}

        for rin_category, price_series in historical_prices.items():
            if rin_category not in self.rin_categories:
                continue

            # Get category specifications
            category_info = self.rin_categories[rin_category]

            # Build forecasting model
            model_data = self._prepare_rin_model_data(
                price_series, compliance_forecast, supply_forecast, rin_category
            )

            if model_data is None or len(model_data) < 30:
                # Insufficient data - use simple trend
                recent_prices = price_series.iloc[-60:]
                trend = recent_prices.diff().mean()
                volatility = recent_prices.std()

                # Simple random walk with trend
                last_price = price_series.iloc[-1]
                forecast_values = []

                for i in range(forecast_horizon):
                    # Add trend and random component
                    random_component = np.random.normal(0, volatility * 0.1)
                    forecast_price = last_price + trend + random_component
                    forecast_values.append(max(0, forecast_price))  # Ensure non-negative

                    last_price = forecast_price

                forecast_dates = pd.date_range(
                    price_series.index[-1] + pd.Timedelta(days=1),
                    periods=forecast_horizon
                )
                forecasts[rin_category] = pd.Series(forecast_values, index=forecast_dates)

            else:
                # Use regression model
                forecast_prices = self._forecast_rin_regression(
                    model_data, forecast_horizon, rin_category
                )

                forecasts[rin_category] = forecast_prices

        return forecasts

    def _prepare_rin_model_data(
        self,
        prices: pd.Series,
        compliance_forecast: Optional[pd.Series],
        supply_forecast: Optional[Dict[str, pd.Series]],
        rin_category: str
    ) -> Optional[pd.DataFrame]:
        """Prepare data for RIN price modeling."""
        # Create feature matrix
        features = pd.DataFrame(index=prices.index)

        # Price lag features
        for lag in [1, 7, 30]:
            features[f'price_lag_{lag}'] = prices.shift(lag)

        # Rolling statistics
        for window in [7, 30, 90]:
            features[f'price_mean_{window}'] = prices.rolling(window).mean()
            features[f'price_std_{window}'] = prices.rolling(window).std()

        # Volume features (if available)
        # In production: Add trading volume data

        # Compliance pressure features
        if compliance_forecast is not None:
            features['compliance_pressure'] = compliance_forecast

            # Compliance trend
            features['compliance_trend'] = compliance_forecast.diff()

        # Supply constraint features
        if supply_forecast and rin_category in supply_forecast:
            supply_data = supply_forecast[rin_category]
            features['supply_level'] = supply_data

            # Supply trend
            features['supply_trend'] = supply_data.diff()

        # Remove NaN values
        features = features.dropna()

        if len(features) < 20:
            return None

        return features

    def _forecast_rin_regression(
        self,
        features: pd.DataFrame,
        horizon: int,
        rin_category: str
    ) -> pd.Series:
        """Generate RIN price forecast using regression model."""
        # Align features with prices
        target_prices = features.index.map(lambda x: self._get_price_at_date(x, rin_category))
        valid_features = features[~target_prices.isna()]
        valid_prices = target_prices[~target_prices.isna()]

        if len(valid_features) < 20:
            # Fallback to simple forecast
            return pd.Series([valid_prices.iloc[-1]] * horizon, index=pd.date_range(
                features.index[-1] + pd.Timedelta(days=1), periods=horizon
            ))

        # Split data for training
        split_point = int(len(valid_features) * 0.8)
        X_train = valid_features.iloc[:split_point]
        y_train = valid_prices.iloc[:split_point]
        X_test = valid_features.iloc[split_point:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Forecast
        forecast_features = features.iloc[-horizon:]
        forecast_scaled = scaler.transform(forecast_features)

        forecast_values = model.predict(forecast_scaled)

        # Create forecast series
        forecast_dates = pd.date_range(
            features.index[-1] + pd.Timedelta(days=1),
            periods=horizon
        )

        return pd.Series(forecast_values, index=forecast_dates)

    def _get_price_at_date(self, date: pd.Timestamp, rin_category: str) -> float:
        """Get RIN price at a specific date (placeholder)."""
        # In production: Query actual price data
        return 1.0  # Placeholder

    def analyze_compliance_demand(
        self,
        gasoline_consumption: pd.Series,
        diesel_consumption: pd.Series,
        ethanol_blend_rates: Optional[pd.Series] = None,
        biodiesel_blend_rates: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze RIN demand from fuel consumption and blending.

        Args:
            gasoline_consumption: Gasoline consumption data
            diesel_consumption: Diesel consumption data
            ethanol_blend_rates: Ethanol blend rates (%)
            biodiesel_blend_rates: Biodiesel blend rates (%)

        Returns:
            Compliance demand analysis
        """
        # Default blend rates if not provided
        if ethanol_blend_rates is None:
            ethanol_blend_rates = pd.Series(0.10, index=gasoline_consumption.index)  # 10% ethanol

        if biodiesel_blend_rates is None:
            biodiesel_blend_rates = pd.Series(0.02, index=diesel_consumption.index)  # 2% biodiesel

        # Calculate RIN generation from blending
        rin_generation = {
            'D6': gasoline_consumption * ethanol_blend_rates * 1.0,  # 1 RIN per gallon ethanol
            'D4': diesel_consumption * biodiesel_blend_rates * 1.5   # 1.5 RINs per gallon biodiesel
        }

        # Calculate total RIN demand (based on total renewable fuel requirements)
        total_gasoline = gasoline_consumption.sum()
        total_diesel = diesel_consumption.sum()

        # RIN demand = (gasoline + diesel) * RIN requirements
        # Simplified: assume 10% renewable requirement
        rin_demand = (total_gasoline + total_diesel) * 0.10

        # D4 vs D6 allocation (simplified)
        d4_demand = rin_demand * 0.15  # 15% for biodiesel
        d6_demand = rin_demand * 0.85  # 85% for ethanol

        return {
            'total_rin_demand': rin_demand,
            'd4_demand': d4_demand,
            'd6_demand': d6_demand,
            'rin_generation': rin_generation,
            'blend_rates_used': {
                'ethanol': ethanol_blend_rates.mean(),
                'biodiesel': biodiesel_blend_rates.mean()
            },
            'total_gasoline_consumption': total_gasoline,
            'total_diesel_consumption': total_diesel
        }

    def model_policy_impact(
        self,
        current_rin_prices: Dict[str, float],
        policy_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Model impact of policy changes on RIN prices.

        Args:
            current_rin_prices: Current RIN prices by category
            policy_scenarios: Policy change scenarios

        Returns:
            Policy impact analysis
        """
        policy_impacts = {}

        for scenario in policy_scenarios:
            scenario_name = scenario['name']
            rin_category = scenario.get('rin_category', 'all')
            demand_impact = scenario.get('demand_impact', 0)  # % change in demand
            supply_impact = scenario.get('supply_impact', 0)  # % change in supply

            impacts = {}

            for category, current_price in current_rin_prices.items():
                if rin_category != 'all' and category != rin_category:
                    continue

                # Simple price impact model
                # Price impact = (demand change - supply change) * price elasticity
                category_info = self.rin_categories.get(category, {})
                price_elasticity = category_info.get('volatility', 0.3)

                net_impact = demand_impact - supply_impact
                price_impact = net_impact * price_elasticity * current_price

                new_price = current_price + price_impact

                impacts[category] = {
                    'current_price': current_price,
                    'price_impact': price_impact,
                    'new_price': new_price,
                    'price_change_pct': (price_impact / current_price) * 100,
                    'demand_impact': demand_impact,
                    'supply_impact': supply_impact
                }

            policy_impacts[scenario_name] = {
                'impacts': impacts,
                'scenario_description': scenario.get('description', ''),
                'categories_affected': list(impacts.keys()),
                'average_price_impact': np.mean([v['price_impact'] for v in impacts.values()]),
                'max_price_impact': max([v['price_impact'] for v in impacts.values()])
            }

        return {
            'policy_scenarios': policy_impacts,
            'current_prices': current_rin_prices,
            'scenarios_analyzed': len(policy_scenarios)
        }

    def calculate_rin_market_equilibrium(
        self,
        supply_curve: Dict[str, float],  # Price -> Supply volume
        demand_curve: Dict[str, float],  # Price -> Demand volume
        price_range: Tuple[float, float] = (0.1, 3.0)
    ) -> Dict[str, float]:
        """
        Calculate RIN market equilibrium price and volume.

        Args:
            supply_curve: Supply at different price levels
            demand_curve: Demand at different price levels
            price_range: Price range for equilibrium search

        Returns:
            Market equilibrium analysis
        """
        # Find equilibrium price where supply = demand
        prices = np.linspace(price_range[0], price_range[1], 100)

        equilibrium_price = None
        equilibrium_volume = None
        min_imbalance = float('inf')

        for price in prices:
            # Interpolate supply and demand at this price
            supply_at_price = self._interpolate_curve(supply_curve, price)
            demand_at_price = self._interpolate_curve(demand_curve, price)

            imbalance = abs(supply_at_price - demand_at_price)

            if imbalance < min_imbalance:
                min_imbalance = imbalance
                equilibrium_price = price
                equilibrium_volume = (supply_at_price + demand_at_price) / 2

        if equilibrium_price is None:
            return {'error': 'No equilibrium found in price range'}

        return {
            'equilibrium_price': equilibrium_price,
            'equilibrium_volume': equilibrium_volume,
            'price_range_tested': price_range,
            'supply_at_equilibrium': supply_at_price,
            'demand_at_equilibrium': demand_at_price,
            'market_balance': supply_at_price / demand_at_price if demand_at_price > 0 else 0
        }

    def _interpolate_curve(self, curve: Dict[str, float], price: float) -> float:
        """Interpolate value from price curve."""
        if price in curve:
            return curve[price]

        # Simple linear interpolation
        prices = sorted(curve.keys())
        if price <= prices[0]:
            return curve[prices[0]]
        elif price >= prices[-1]:
            return curve[prices[-1]]

        # Find interpolation points
        for i in range(len(prices) - 1):
            if prices[i] <= price <= prices[i + 1]:
                price1, price2 = prices[i], prices[i + 1]
                value1, value2 = curve[price1], curve[price2]

                # Linear interpolation
                return value1 + (value2 - value1) * (price - price1) / (price2 - price1)

        return 0  # Fallback

    def analyze_rin_volatility_regimes(
        self,
        rin_prices: Dict[str, pd.Series],
        regime_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze RIN price volatility regimes.

        Args:
            rin_prices: Historical RIN price data by category
            regime_thresholds: Thresholds for regime classification

        Returns:
            Volatility regime analysis
        """
        if regime_thresholds is None:
            regime_thresholds = {
                'low_vol': 0.05,    # 5% daily volatility
                'medium_vol': 0.15, # 15% daily volatility
                'high_vol': 0.30    # 30% daily volatility
            }

        regime_analysis = {}

        for rin_category, price_series in rin_prices.items():
            # Calculate daily returns
            returns = price_series.pct_change().dropna()

            if len(returns) < 30:
                continue

            # Calculate rolling volatility
            volatility = returns.rolling(30).std()

            # Classify volatility regimes
            regime_classification = pd.Series(index=volatility.index)

            for date, vol in volatility.items():
                if vol <= regime_thresholds['low_vol']:
                    regime_classification[date] = 'low_volatility'
                elif vol <= regime_thresholds['medium_vol']:
                    regime_classification[date] = 'medium_volatility'
                else:
                    regime_classification[date] = 'high_volatility'

            # Regime statistics
            regime_stats = regime_classification.value_counts()

            regime_analysis[rin_category] = {
                'average_volatility': volatility.mean(),
                'max_volatility': volatility.max(),
                'regime_distribution': regime_stats.to_dict(),
                'regime_classification': regime_classification,
                'low_volatility_days': regime_stats.get('low_volatility', 0),
                'high_volatility_days': regime_stats.get('high_volatility', 0)
            }

        return {
            'regime_analysis': regime_analysis,
            'regime_thresholds': regime_thresholds,
            'categories_analyzed': list(rin_prices.keys())
        }

    def forecast_rin_supply_demand_balance(
        self,
        rin_generation: Dict[str, pd.Series],
        rin_demand: Dict[str, pd.Series],
        forecast_horizon: int = 90
    ) -> Dict[str, pd.Series]:
        """
        Forecast RIN supply-demand balance.

        Args:
            rin_generation: RIN generation by category
            rin_demand: RIN demand by category
            forecast_horizon: Days to forecast

        Returns:
            Supply-demand balance forecasts
        """
        balance_forecasts = {}

        for category in rin_generation.keys():
            if category not in rin_demand:
                continue

            generation = rin_generation[category]
            demand = rin_demand[category]

            # Align data
            combined_data = pd.DataFrame({
                'generation': generation,
                'demand': demand
            }).dropna()

            if len(combined_data) < 20:
                continue

            # Calculate historical balance
            balance = combined_data['generation'] - combined_data['demand']

            # Simple forecasting (trend + seasonal)
            # In production: Use more sophisticated time series models

            # Calculate trend
            balance_trend = balance.diff().mean()

            # Calculate seasonal pattern
            monthly_balance = balance.groupby(balance.index.month).mean()

            # Generate forecast
            last_date = balance.index[-1]
            forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)

            forecast_values = []
            for i in range(forecast_horizon):
                forecast_date = forecast_dates[i]
                month = forecast_date.month

                # Base forecast = last balance + trend
                base_forecast = balance.iloc[-1] + balance_trend * (i + 1)

                # Add seasonal adjustment
                if month in monthly_balance.index:
                    seasonal_adjustment = monthly_balance[month]
                else:
                    seasonal_adjustment = 0

                forecast_value = base_forecast + seasonal_adjustment
                forecast_values.append(forecast_value)

            balance_forecasts[category] = pd.Series(forecast_values, index=forecast_dates)

        return balance_forecasts

    def calculate_rin_portfolio_hedging(
        self,
        current_positions: Dict[str, float],
        price_forecasts: Dict[str, pd.Series],
        risk_tolerance: float = 0.05,  # 5% risk tolerance
        hedge_ratio: float = 0.8       # 80% hedge ratio
    ) -> Dict[str, Any]:
        """
        Calculate optimal RIN portfolio hedging strategy.

        Args:
            current_positions: Current RIN positions by category
            price_forecasts: Price forecasts by category
            risk_tolerance: Maximum acceptable risk (%)
            hedge_ratio: Target hedge ratio

        Returns:
            Hedging strategy recommendations
        """
        hedging_strategy = {}

        for category, position in current_positions.items():
            if category not in price_forecasts:
                continue

            price_forecast = price_forecasts[category]

            # Calculate expected price and volatility
            expected_price = price_forecast.mean()
            price_volatility = price_forecast.std()

            # Calculate position value and risk
            position_value = position * expected_price
            position_risk = position * price_volatility

            # Calculate hedge requirement
            target_hedge = position * hedge_ratio
            current_exposure = position  # Assuming no current hedge

            hedge_adjustment = target_hedge - current_exposure

            # Risk assessment
            risk_exposure = position_risk / position_value if position_value > 0 else 0
            within_risk_tolerance = risk_exposure <= risk_tolerance

            hedging_strategy[category] = {
                'position': position,
                'expected_price': expected_price,
                'position_value': position_value,
                'position_risk': position_risk,
                'risk_exposure': risk_exposure,
                'target_hedge': target_hedge,
                'hedge_adjustment': hedge_adjustment,
                'within_risk_tolerance': within_risk_tolerance,
                'recommended_action': 'increase_hedge' if hedge_adjustment > 0 else 'decrease_hedge'
            }

        # Portfolio-level analysis
        total_position_value = sum(strategy['position_value'] for strategy in hedging_strategy.values())
        total_risk = sum(strategy['position_risk'] for strategy in hedging_strategy.values())
        portfolio_risk = total_risk / total_position_value if total_position_value > 0 else 0

        return {
            'hedging_strategy': hedging_strategy,
            'portfolio_risk': portfolio_risk,
            'total_position_value': total_position_value,
            'risk_tolerance': risk_tolerance,
            'hedge_ratio': hedge_ratio,
            'portfolio_within_tolerance': portfolio_risk <= risk_tolerance
        }
