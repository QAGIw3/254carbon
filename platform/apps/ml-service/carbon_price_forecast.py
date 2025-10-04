"""
Carbon Price Forecasting Model

Advanced model for forecasting carbon allowance and credit prices:
- EUA and CCA price forecasting
- RIN price forecasting
- REC price forecasting
- Policy impact modeling
- Market sentiment analysis
- Supply-demand balance forecasting
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class CarbonPriceForecast:
    """
    Carbon price forecasting and analysis model.

    Features:
    - Multi-market carbon price forecasting
    - Policy scenario modeling
    - Supply-demand balance analysis
    - Market sentiment integration
    """

    def __init__(self):
        # Carbon market fundamentals
        self.carbon_markets = {
            'eua': {
                'market': 'EU ETS',
                'compliance_period': 'Phase IV (2021-2030)',
                'cap_decline': 0.024,  # 2.4% annual cap reduction
                'auction_volume': 500000000,  # 500M allowances per year
                'volatility': 0.25
            },
            'cca': {
                'market': 'California Cap-and-Trade',
                'compliance_period': '2021-2025',
                'cap_decline': 0.03,  # 3% annual cap reduction
                'auction_volume': 200000000,  # 200M allowances per year
                'volatility': 0.30
            },
            'rggi': {
                'market': 'Regional Greenhouse Gas Initiative',
                'compliance_period': '2021-2025',
                'cap_decline': 0.03,  # 3% annual cap reduction
                'auction_volume': 50000000,  # 50M allowances per year
                'volatility': 0.35
            }
        }

        # Policy impact factors
        self.policy_factors = {
            'eua': {
                'msr_activation': 0.12,  # Market Stability Reserve impact
                'cbam_introduction': 0.08,  # Carbon Border Adjustment Mechanism
                'renewable_targets': 0.05   # EU renewable energy targets
            },
            'cca': {
                'cap_reduction': 0.15,     # Stricter cap reductions
                'transport_inclusion': 0.10,  # Transportation sector inclusion
                'offset_limits': 0.08      # Offset credit restrictions
            },
            'rggi': {
                'state_expansion': 0.10,   # Potential state additions
                'emissions_targets': 0.12, # State emissions targets
                'offset_usage': 0.05       # Offset credit usage
            }
        }

    def forecast_carbon_prices(
        self,
        historical_prices: Dict[str, pd.Series],
        policy_scenarios: Optional[List[Dict[str, Any]]] = None,
        forecast_horizon: int = 365,
        include_policy_impact: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Forecast carbon prices for multiple markets.

        Args:
            historical_prices: Historical price data by market
            policy_scenarios: Policy change scenarios
            forecast_horizon: Days to forecast
            include_policy_impact: Include policy impact modeling

        Returns:
            Carbon price forecasts by market
        """
        forecasts = {}

        for market, price_series in historical_prices.items():
            if market not in self.carbon_markets:
                continue

            # Build forecasting model
            model_data = self._prepare_carbon_model_data(
                price_series, market, policy_scenarios
            )

            if model_data is None or len(model_data) < 50:
                # Insufficient data - use trend-based forecast
                forecast_prices = self._trend_based_forecast(
                    price_series, forecast_horizon, market
                )
            else:
                # Use regression model
                forecast_prices = self._regression_forecast(
                    model_data, forecast_horizon, market
                )

            forecasts[market] = forecast_prices

        return forecasts

    def _prepare_carbon_model_data(
        self,
        prices: pd.Series,
        market: str,
        policy_scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[pd.DataFrame]:
        """Prepare data for carbon price modeling."""
        # Create feature matrix
        features = pd.DataFrame(index=prices.index)

        # Price lag features
        for lag in [1, 7, 30, 90]:
            features[f'price_lag_{lag}'] = prices.shift(lag)

        # Rolling statistics
        for window in [7, 30, 90]:
            features[f'price_mean_{window}'] = prices.rolling(window).mean()
            features[f'price_std_{window}'] = prices.rolling(window).std()

        # Technical indicators
        features['price_momentum'] = (prices - prices.shift(30)) / prices.shift(30)
        features['price_volatility'] = prices.rolling(30).std() / prices.rolling(30).mean()

        # Market fundamentals
        market_info = self.carbon_markets[market]
        features['cap_decline_rate'] = market_info['cap_decline']
        features['auction_volume'] = market_info['auction_volume']

        # Policy impact features
        if policy_scenarios:
            for scenario in policy_scenarios:
                scenario_name = scenario['name']
                impact = scenario.get('price_impact', 0)
                features[f'policy_{scenario_name}'] = impact

        # Remove NaN values
        features = features.dropna()

        if len(features) < 20:
            return None

        return features

    def _trend_based_forecast(
        self,
        prices: pd.Series,
        horizon: int,
        market: str
    ) -> pd.Series:
        """Generate trend-based carbon price forecast."""
        # Calculate trend and volatility
        recent_prices = prices.iloc[-90:]  # Last 90 days
        trend = recent_prices.diff().mean()
        volatility = recent_prices.std()

        # Generate forecast
        last_price = prices.iloc[-1]
        forecast_values = []

        for i in range(horizon):
            # Add trend and random component
            random_component = np.random.normal(0, volatility * 0.1)
            forecast_price = last_price + trend + random_component
            forecast_values.append(max(0, forecast_price))

            last_price = forecast_price

        # Create forecast series
        forecast_dates = pd.date_range(
            prices.index[-1] + pd.Timedelta(days=1),
            periods=horizon
        )

        return pd.Series(forecast_values, index=forecast_dates)

    def _regression_forecast(
        self,
        features: pd.DataFrame,
        horizon: int,
        market: str
    ) -> pd.Series:
        """Generate carbon price forecast using regression model."""
        # Align features with prices
        target_prices = features.index.map(lambda x: self._get_price_at_date(x, market))
        valid_features = features[~target_prices.isna()]
        valid_prices = target_prices[~target_prices.isna()]

        if len(valid_features) < 20:
            # Fallback to trend forecast
            return self._trend_based_forecast(
                pd.Series(target_prices, index=features.index), horizon, market
            )

        # Split data for training
        split_point = int(len(valid_features) * 0.8)
        X_train = valid_features.iloc[:split_point]
        y_train = valid_prices.iloc[:split_point]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
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

    def _get_price_at_date(self, date: pd.Timestamp, market: str) -> float:
        """Get carbon price at a specific date (placeholder)."""
        # In production: Query actual price data
        return 50.0  # Placeholder

    def analyze_compliance_cost_impact(
        self,
        carbon_prices: Dict[str, pd.Series],
        emissions_data: Dict[str, pd.Series],
        compliance_obligations: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze compliance cost impact on different sectors.

        Args:
            carbon_prices: Carbon price forecasts by market
            emissions_data: Historical emissions by sector
            compliance_obligations: Compliance obligations by sector

        Returns:
            Compliance cost impact analysis
        """
        compliance_costs = {}

        for market, price_series in carbon_prices.items():
            if market not in emissions_data:
                continue

            emissions = emissions_data[market]

            # Calculate compliance costs
            avg_price = price_series.mean()
            total_emissions = emissions.sum()

            # Cost per tonne
            cost_per_tonne = avg_price

            # Total compliance cost
            total_cost = total_emissions * cost_per_tonne

            # Cost by sector (if sector data available)
            sector_costs = {}
            for sector, obligation in compliance_obligations.items():
                sector_emissions = emissions * (obligation / sum(compliance_obligations.values()))
                sector_costs[sector] = sector_emissions.sum() * cost_per_tonne

            compliance_costs[market] = {
                'total_compliance_cost': total_cost,
                'cost_per_tonne': cost_per_tonne,
                'total_emissions': total_emissions,
                'sector_costs': sector_costs,
                'average_price': avg_price,
                'price_volatility': price_series.std()
            }

        return {
            'compliance_costs': compliance_costs,
            'total_system_cost': sum(
                cost['total_compliance_cost'] for cost in compliance_costs.values()
            ),
            'markets_analyzed': len(compliance_costs),
            'sectors_affected': list(compliance_obligations.keys())
        }

    def model_carbon_leakage_risk(
        self,
        domestic_carbon_prices: Dict[str, float],
        international_competitor_prices: Dict[str, float],
        trade_exposure: Dict[str, float],
        emissions_intensity: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Model risk of carbon leakage to less-regulated jurisdictions.

        Args:
            domestic_carbon_prices: Carbon prices in domestic markets
            international_competitor_prices: Carbon prices in competitor countries
            trade_exposure: Trade exposure by sector
            emissions_intensity: Emissions intensity by sector

        Returns:
            Carbon leakage risk analysis
        """
        leakage_risks = {}

        for sector in trade_exposure.keys():
            domestic_price = domestic_carbon_prices.get(sector, 0)
            international_price = international_competitor_prices.get(sector, 0)

            # Calculate price differential
            price_differential = domestic_price - international_price

            # Calculate leakage risk
            trade_exposure_pct = trade_exposure[sector]
            emissions_intensity_val = emissions_intensity.get(sector, 1.0)

            # Leakage risk score (0-1 scale)
            leakage_risk = min(1.0, (price_differential * trade_exposure_pct * emissions_intensity_val) / 100)

            leakage_risks[sector] = {
                'domestic_carbon_price': domestic_price,
                'international_carbon_price': international_price,
                'price_differential': price_differential,
                'trade_exposure': trade_exposure_pct,
                'emissions_intensity': emissions_intensity_val,
                'leakage_risk_score': leakage_risk,
                'risk_level': 'high' if leakage_risk > 0.7 else ('medium' if leakage_risk > 0.4 else 'low')
            }

        # Overall leakage risk
        avg_leakage_risk = np.mean([risk['leakage_risk_score'] for risk in leakage_risks.values()])

        return {
            'leakage_risks': leakage_risks,
            'average_leakage_risk': avg_leakage_risk,
            'high_risk_sectors': [
                sector for sector, risk in leakage_risks.items()
                if risk['risk_level'] == 'high'
            ],
            'overall_risk_level': 'high' if avg_leakage_risk > 0.7 else ('medium' if avg_leakage_risk > 0.4 else 'low'),
            'sectors_analyzed': len(leakage_risks)
        }

    def forecast_policy_scenario_impact(
        self,
        baseline_prices: Dict[str, float],
        policy_scenarios: List[Dict[str, Any]],
        scenario_probabilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Forecast impact of policy scenarios on carbon prices.

        Args:
            baseline_prices: Current carbon prices by market
            policy_scenarios: Policy change scenarios
            scenario_probabilities: Probability of each scenario

        Returns:
            Policy scenario impact analysis
        """
        if scenario_probabilities is None:
            # Equal probability for all scenarios
            scenario_probabilities = {scenario['name']: 1/len(policy_scenarios) for scenario in policy_scenarios}

        scenario_impacts = {}

        for scenario in policy_scenarios:
            scenario_name = scenario['name']
            probability = scenario_probabilities.get(scenario_name, 0)

            impacts = {}

            for market, baseline_price in baseline_prices.items():
                # Calculate price impact for this scenario
                price_impact = scenario.get('price_impact', 0) * baseline_price

                # Apply market-specific factors
                market_factor = scenario.get(f'{market}_factor', 1.0)
                adjusted_impact = price_impact * market_factor

                new_price = baseline_price + adjusted_impact

                impacts[market] = {
                    'baseline_price': baseline_price,
                    'price_impact': adjusted_impact,
                    'new_price': new_price,
                    'price_change_pct': (adjusted_impact / baseline_price) * 100,
                    'scenario_probability': probability
                }

            scenario_impacts[scenario_name] = {
                'impacts': impacts,
                'scenario_probability': probability,
                'expected_price_impact': sum(
                    impact['price_impact'] * probability
                    for impact in impacts.values()
                ),
                'scenario_description': scenario.get('description', '')
            }

        # Calculate expected prices across all scenarios
        expected_prices = {}
        for market in baseline_prices.keys():
            expected_price = sum(
                scenario['impacts'][market]['new_price'] * scenario['scenario_probability']
                for scenario in scenario_impacts.values()
            )
            expected_prices[market] = expected_price

        return {
            'policy_scenarios': scenario_impacts,
            'expected_prices': expected_prices,
            'baseline_prices': baseline_prices,
            'scenario_probabilities': scenario_probabilities,
            'risk_adjusted_prices': expected_prices  # Same as expected in this simple model
        }

    def analyze_market_sentiment_impact(
        self,
        price_data: pd.Series,
        news_sentiment: Optional[pd.Series] = None,
        trading_volume: Optional[pd.Series] = None,
        social_media_sentiment: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze impact of market sentiment on carbon prices.

        Args:
            price_data: Carbon price data
            news_sentiment: News sentiment scores (-1 to 1)
            trading_volume: Trading volume data
            social_media_sentiment: Social media sentiment scores

        Returns:
            Sentiment impact analysis
        """
        # Align data
        sentiment_data = pd.DataFrame({
            'price': price_data,
            'news_sentiment': news_sentiment,
            'trading_volume': trading_volume,
            'social_sentiment': social_media_sentiment
        }).dropna()

        if len(sentiment_data) < 30:
            return {'error': 'Insufficient data for sentiment analysis'}

        # Calculate correlations
        correlations = {}
        for col in ['news_sentiment', 'trading_volume', 'social_sentiment']:
            if col in sentiment_data.columns:
                corr = np.corrcoef(sentiment_data['price'], sentiment_data[col])[0, 1]
                correlations[col] = corr

        # Analyze sentiment impact on price movements
        sentiment_impact = {}

        if 'news_sentiment' in sentiment_data.columns:
            # Positive news sentiment tends to increase prices
            positive_news = sentiment_data[sentiment_data['news_sentiment'] > 0.1]
            negative_news = sentiment_data[sentiment_data['news_sentiment'] < -0.1]

            if len(positive_news) > 0 and len(negative_news) > 0:
                positive_price_change = (positive_news['price'] - positive_news['price'].shift(1)).mean()
                negative_price_change = (negative_news['price'] - negative_news['price'].shift(1)).mean()

                sentiment_impact['news'] = {
                    'positive_price_impact': positive_price_change,
                    'negative_price_impact': negative_price_change,
                    'sentiment_effectiveness': abs(positive_price_change) > abs(negative_price_change)
                }

        return {
            'correlations': correlations,
            'sentiment_impact': sentiment_impact,
            'data_points': len(sentiment_data),
            'sentiment_sources': list(correlations.keys())
        }

    def calculate_carbon_price_risk_metrics(
        self,
        price_forecasts: Dict[str, pd.Series],
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate risk metrics for carbon price forecasts.

        Args:
            price_forecasts: Carbon price forecasts by market
            confidence_level: Confidence level for VaR calculation

        Returns:
            Risk metrics by market
        """
        risk_metrics = {}

        for market, forecast_series in price_forecasts.items():
            if len(forecast_series) < 10:
                continue

            # Basic statistics
            mean_price = forecast_series.mean()
            std_price = forecast_series.std()

            # Value at Risk (VaR)
            var = forecast_series.quantile(1 - confidence_level)

            # Conditional Value at Risk (CVaR)
            cvar = forecast_series[forecast_series <= var].mean()

            # Maximum drawdown
            cumulative = forecast_series.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratio = mean_price / std_price if std_price > 0 else 0

            risk_metrics[market] = {
                'mean_price': mean_price,
                'price_volatility': std_price,
                'value_at_risk': var,
                'conditional_var': cvar,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'confidence_level': confidence_level,
                'forecast_horizon': len(forecast_series)
            }

        return risk_metrics

    def optimize_carbon_hedging_strategy(
        self,
        current_exposure: Dict[str, float],
        price_forecasts: Dict[str, pd.Series],
        risk_tolerance: float = 0.05,
        hedge_instruments: List[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize carbon hedging strategy.

        Args:
            current_exposure: Current carbon exposure by market
            price_forecasts: Price forecasts by market
            risk_tolerance: Maximum acceptable risk
            hedge_instruments: Available hedge instruments

        Returns:
            Optimal hedging strategy
        """
        if hedge_instruments is None:
            hedge_instruments = ['futures', 'options', 'swaps']

        hedging_strategy = {}

        for market, exposure in current_exposure.items():
            if market not in price_forecasts:
                continue

            forecast = price_forecasts[market]

            # Calculate exposure value and risk
            expected_price = forecast.mean()
            price_volatility = forecast.std()

            exposure_value = exposure * expected_price
            exposure_risk = exposure * price_volatility

            # Calculate hedge requirements
            target_hedge_ratio = min(1.0, risk_tolerance / (exposure_risk / exposure_value) if exposure_value > 0 else 0)

            hedging_strategy[market] = {
                'current_exposure': exposure,
                'exposure_value': exposure_value,
                'exposure_risk': exposure_risk,
                'target_hedge_ratio': target_hedge_ratio,
                'hedge_volume': exposure * target_hedge_ratio,
                'available_instruments': hedge_instruments,
                'risk_reduction': exposure_risk * (1 - target_hedge_ratio),
                'expected_cost': exposure * target_hedge_ratio * expected_price * 0.02  # 2% hedge cost
            }

        # Portfolio-level analysis
        total_exposure_value = sum(strategy['exposure_value'] for strategy in hedging_strategy.values())
        total_risk = sum(strategy['exposure_risk'] for strategy in hedging_strategy.values())

        return {
            'hedging_strategy': hedging_strategy,
            'portfolio_risk': total_risk / total_exposure_value if total_exposure_value > 0 else 0,
            'total_hedge_cost': sum(strategy['expected_cost'] for strategy in hedging_strategy.values()),
            'markets_hedged': len(hedging_strategy),
            'risk_tolerance': risk_tolerance
        }
