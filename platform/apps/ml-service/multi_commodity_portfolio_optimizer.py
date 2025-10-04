"""
Multi-Commodity Portfolio Optimization Model

Advanced model for optimizing portfolios across multiple energy commodities:
- Cross-commodity correlation analysis
- Risk-adjusted return optimization
- Diversification strategies
- Dynamic rebalancing algorithms
- Performance attribution analysis
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.covariance import LedoitWolf, OAS
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MultiCommodityPortfolioOptimizer:
    """
    Multi-commodity portfolio optimization and analysis.

    Features:
    - Modern portfolio theory application to commodities
    - Risk parity strategies
    - Dynamic rebalancing
    - Performance attribution
    """

    def __init__(self):
        self.portfolio_history = {}
        self.risk_models = {}

    def optimize_portfolio_weights(
        self,
        returns_data: pd.DataFrame,
        risk_free_rate: float = 0.02,
        target_return: Optional[float] = None,
        max_weight: float = 0.3,
        optimization_method: str = 'mean_variance'
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights using modern portfolio theory.

        Args:
            returns_data: Historical returns by commodity
            risk_free_rate: Risk-free rate for Sharpe ratio
            target_return: Target portfolio return (optional)
            max_weight: Maximum weight per commodity
            optimization_method: 'mean_variance', 'risk_parity', or 'equal_weight'

        Returns:
            Optimal portfolio weights and analysis
        """
        # Calculate expected returns and covariance
        expected_returns = returns_data.mean() * 252  # Annualized
        covariance_matrix = returns_data.cov() * 252   # Annualized

        commodities = returns_data.columns.tolist()

        if optimization_method == 'mean_variance':
            # Mean-variance optimization
            optimal_weights = self._mean_variance_optimization(
                expected_returns, covariance_matrix, target_return, max_weight
            )

        elif optimization_method == 'risk_parity':
            # Risk parity optimization
            optimal_weights = self._risk_parity_optimization(
                covariance_matrix, commodities, max_weight
            )

        elif optimization_method == 'equal_weight':
            # Equal weight portfolio
            n_commodities = len(commodities)
            optimal_weights = {commodity: 1/n_commodities for commodity in commodities}

        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            optimal_weights, expected_returns, covariance_matrix, risk_free_rate
        )

        return {
            'optimal_weights': optimal_weights,
            'portfolio_metrics': portfolio_metrics,
            'expected_returns': expected_returns,
            'covariance_matrix': covariance_matrix,
            'optimization_method': optimization_method,
            'commodities': commodities,
            'rebalancing_frequency': 'monthly'  # Recommended rebalancing
        }

    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        target_return: Optional[float],
        max_weight: float
    ) -> Dict[str, float]:
        """Perform mean-variance optimization."""
        commodities = expected_returns.index.tolist()
        n = len(commodities)

        # Define objective function (minimize risk)
        def objective(weights):
            portfolio_risk = np.dot(weights.T, np.dot(covariance_matrix.values, weights))
            return portfolio_risk

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        if target_return is not None:
            # Target return constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns.values) - target_return
            })

        # Bounds (0 to max_weight for each commodity)
        bounds = [(0, max_weight) for _ in range(n)]

        # Initial guess (equal weights)
        initial_weights = np.array([1/n] * n)

        # Optimize
        result = optimize.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Fallback to equal weights
            optimal_weights = {commodity: 1/n for commodity in commodities}
        else:
            optimal_weights = dict(zip(commodities, result.x))

        return optimal_weights

    def _risk_parity_optimization(
        self,
        covariance_matrix: pd.DataFrame,
        commodities: List[str],
        max_weight: float
    ) -> Dict[str, float]:
        """Perform risk parity optimization."""
        # Risk parity: equal risk contribution from each asset

        # Calculate inverse variance weights
        variances = np.diag(covariance_matrix.values)
        inv_variances = 1 / variances

        # Normalize to sum to 1
        total_inv_var = np.sum(inv_variances)
        risk_parity_weights = inv_variances / total_inv_var

        # Apply max weight constraint
        risk_parity_weights = np.minimum(risk_parity_weights, max_weight)
        risk_parity_weights = risk_parity_weights / np.sum(risk_parity_weights)

        return dict(zip(commodities, risk_parity_weights))

    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        # Portfolio expected return
        portfolio_return = sum(weights[commodity] * expected_returns[commodity]
                              for commodity in weights.keys())

        # Portfolio risk (standard deviation)
        weight_array = np.array([weights[commodity] for commodity in expected_returns.index])
        portfolio_risk = np.sqrt(np.dot(weight_array.T, np.dot(covariance_matrix.values, weight_array)))

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

        # Maximum drawdown (simplified calculation)
        # In production: Use more sophisticated drawdown calculation
        max_drawdown = 0.15  # Placeholder

        # Diversification ratio
        weighted_risks = [weights[commodity] * np.sqrt(covariance_matrix.loc[commodity, commodity])
                         for commodity in weights.keys()]
        total_weighted_risk = sum(weighted_risks)
        average_correlation = 0.3  # Placeholder

        diversification_ratio = portfolio_risk / total_weighted_risk if total_weighted_risk > 0 else 1

        return {
            'expected_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'diversification_ratio': diversification_ratio,
            'risk_free_rate': risk_free_rate,
            'total_weight': sum(weights.values())
        }

    def detect_cross_market_arbitrage(
        self,
        price_data: Dict[str, pd.Series],
        transport_costs: Dict[str, Dict[str, float]] = None,
        storage_costs: Dict[str, float] = None,
        arbitrage_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect arbitrage opportunities across commodity markets.

        Args:
            price_data: Price data by commodity and location
            transport_costs: Transport costs between locations
            storage_costs: Storage costs by commodity
            arbitrage_threshold: Minimum profit threshold for arbitrage

        Returns:
            Arbitrage opportunity analysis
        """
        if transport_costs is None:
            transport_costs = {}
        if storage_costs is None:
            storage_costs = {commodity: 0 for commodity in price_data.keys()}

        arbitrage_opportunities = []

        commodities = list(price_data.keys())

        for i, commodity1 in enumerate(commodities[:-1]):
            for commodity2 in commodities[i+1:]:
                # Get price data for both commodities
                prices1 = price_data[commodity1]
                prices2 = price_data[commodity2]

                # Find common time periods
                common_index = prices1.index.intersection(prices2.index)

                if len(common_index) < 10:
                    continue

                # Calculate price spreads
                spreads = prices1.loc[common_index] - prices2.loc[common_index]

                # Calculate spread statistics
                mean_spread = spreads.mean()
                spread_volatility = spreads.std()

                # Check for arbitrage opportunities
                if abs(mean_spread) > arbitrage_threshold:
                    # Calculate transport costs
                    transport_cost = transport_costs.get(commodity1, {}).get(commodity2, 0)

                    # Calculate storage costs
                    storage_cost = (storage_costs.get(commodity1, 0) + storage_costs.get(commodity2, 0)) / 2

                    # Net arbitrage profit
                    net_profit = abs(mean_spread) - transport_cost - storage_cost

                    if net_profit > arbitrage_threshold:
                        arbitrage_opportunities.append({
                            'commodity1': commodity1,
                            'commodity2': commodity2,
                            'mean_spread': mean_spread,
                            'spread_volatility': spread_volatility,
                            'transport_cost': transport_cost,
                            'storage_cost': storage_cost,
                            'net_profit': net_profit,
                            'arbitrage_direction': 'buy_cheap_sell_expensive' if mean_spread > 0 else 'buy_expensive_sell_cheap',
                            'confidence_score': min(1.0, net_profit / arbitrage_threshold),
                            'opportunity_period': f'{common_index.min().date()} to {common_index.max().date()}'
                        })

        # Sort by profitability
        arbitrage_opportunities.sort(key=lambda x: x['net_profit'], reverse=True)

        return {
            'arbitrage_opportunities': arbitrage_opportunities,
            'total_opportunities': len(arbitrage_opportunities),
            'best_opportunity': arbitrage_opportunities[0] if arbitrage_opportunities else None,
            'total_potential_profit': sum(opp['net_profit'] for opp in arbitrage_opportunities),
            'arbitrage_threshold': arbitrage_threshold,
            'markets_analyzed': len(commodities)
        }

    def calculate_integrated_risk_metrics(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate integrated risk metrics for multi-commodity portfolio.

        Args:
            portfolio_weights: Portfolio weights by commodity
            returns_data: Historical returns data
            correlation_matrix: Correlation matrix between commodities

        Returns:
            Integrated risk metrics
        """
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=returns_data.index)

        for commodity, weight in portfolio_weights.items():
            if commodity in returns_data.columns:
                portfolio_returns += weight * returns_data[commodity]

        # Basic risk metrics
        portfolio_mean = portfolio_returns.mean()
        portfolio_std = portfolio_returns.std()
        portfolio_var = portfolio_returns.var()

        # Value at Risk (95% confidence)
        var_95 = portfolio_returns.quantile(0.05)

        # Conditional Value at Risk (Expected Shortfall)
        tail_losses = portfolio_returns[portfolio_returns <= var_95]
        cvar_95 = tail_losses.mean() if len(tail_losses) > 0 else var_95

        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Beta to benchmark (if provided)
        beta = 1.0  # Placeholder - would calculate vs energy benchmark

        # Diversification benefit
        individual_risks = [
            portfolio_weights[commodity] * returns_data[commodity].std()
            for commodity in portfolio_weights.keys()
            if commodity in returns_data.columns
        ]

        total_individual_risk = sum(individual_risks)
        diversification_benefit = (total_individual_risk - portfolio_std) / total_individual_risk if total_individual_risk > 0 else 0

        return {
            'portfolio_mean_return': portfolio_mean,
            'portfolio_volatility': portfolio_std,
            'portfolio_variance': portfolio_var,
            'value_at_risk_95': var_95,
            'conditional_var_95': cvar_95,
            'maximum_drawdown': max_drawdown,
            'beta': beta,
            'diversification_benefit': diversification_benefit,
            'sharpe_ratio': portfolio_mean / portfolio_std if portfolio_std > 0 else 0,
            'sortino_ratio': portfolio_mean / (portfolio_returns[portfolio_returns < 0].std()) if (portfolio_returns < 0).std() > 0 else 0
        }

    def perform_scenario_stress_testing(
        self,
        portfolio_weights: Dict[str, float],
        scenario_definitions: List[Dict[str, Any]],
        base_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform stress testing under various market scenarios.

        Args:
            portfolio_weights: Portfolio weights by commodity
            scenario_definitions: Market stress scenarios
            base_returns: Historical returns data

        Returns:
            Stress testing results
        """
        stress_results = {}

        for scenario in scenario_definitions:
            scenario_name = scenario['name']

            # Apply scenario shocks to returns
            scenario_returns = base_returns.copy()

            for shock in scenario.get('shocks', []):
                commodity = shock['commodity']
                shock_type = shock['type']
                shock_magnitude = shock['magnitude']

                if commodity in scenario_returns.columns:
                    if shock_type == 'absolute':
                        # Absolute return shock
                        scenario_returns[commodity] += shock_magnitude
                    elif shock_type == 'relative':
                        # Relative return shock
                        scenario_returns[commodity] *= (1 + shock_magnitude)
                    elif shock_type == 'volatility':
                        # Volatility shock
                        scenario_returns[commodity] *= (1 + shock_magnitude * np.random.randn(len(scenario_returns)))

            # Calculate portfolio performance under scenario
            portfolio_scenario_returns = pd.Series(0, index=scenario_returns.index)

            for commodity, weight in portfolio_weights.items():
                if commodity in scenario_returns.columns:
                    portfolio_scenario_returns += weight * scenario_returns[commodity]

            # Calculate scenario metrics
            scenario_mean = portfolio_scenario_returns.mean()
            scenario_volatility = portfolio_scenario_returns.std()
            scenario_max_loss = portfolio_scenario_returns.min()

            # Compare to baseline
            baseline_mean = (base_returns * pd.Series(portfolio_weights)).sum(axis=1).mean()
            baseline_volatility = (base_returns * pd.Series(portfolio_weights)).sum(axis=1).std()

            stress_results[scenario_name] = {
                'scenario_mean_return': scenario_mean,
                'scenario_volatility': scenario_volatility,
                'scenario_max_loss': scenario_max_loss,
                'return_impact': scenario_mean - baseline_mean,
                'volatility_impact': scenario_volatility - baseline_volatility,
                'scenario_severity': scenario.get('severity', 'moderate'),
                'scenario_probability': scenario.get('probability', 0.05),
                'risk_adjusted_impact': (scenario_mean - baseline_mean) / scenario_volatility if scenario_volatility > 0 else 0
            }

        # Portfolio stress summary
        avg_return_impact = np.mean([result['return_impact'] for result in stress_results.values()])
        avg_volatility_impact = np.mean([result['volatility_impact'] for result in stress_results.values()])

        return {
            'stress_test_results': stress_results,
            'portfolio_stress_summary': {
                'average_return_impact': avg_return_impact,
                'average_volatility_impact': avg_volatility_impact,
                'worst_case_scenario': min(stress_results.items(), key=lambda x: x[1]['scenario_max_loss'])[0],
                'best_case_scenario': max(stress_results.items(), key=lambda x: x[1]['scenario_mean_return'])[0]
            },
            'scenarios_tested': len(stress_results),
            'portfolio_weights': portfolio_weights
        }

    def calculate_performance_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        commodity_returns: pd.DataFrame,
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate performance attribution for multi-commodity portfolio.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            commodity_returns: Individual commodity returns
            portfolio_weights: Portfolio weights

        Returns:
            Performance attribution analysis
        """
        # Excess return vs benchmark
        excess_returns = portfolio_returns - benchmark_returns

        # Attribution by commodity
        commodity_attribution = {}

        for commodity in portfolio_weights.keys():
            if commodity in commodity_returns.columns:
                # Calculate commodity contribution
                commodity_contribution = portfolio_weights[commodity] * commodity_returns[commodity]

                # Correlation with portfolio
                correlation = np.corrcoef(portfolio_returns, commodity_contribution)[0, 1]

                commodity_attribution[commodity] = {
                    'weight': portfolio_weights[commodity],
                    'contribution': commodity_contribution.mean(),
                    'correlation': correlation,
                    'attribution_impact': portfolio_weights[commodity] * commodity_returns[commodity].mean(),
                    'selection_effect': (commodity_returns[commodity] - benchmark_returns).mean() * portfolio_weights[commodity],
                    'allocation_effect': (benchmark_returns - benchmark_returns.mean()).mean() * portfolio_weights[commodity]
                }

        # Calculate total attribution effects
        total_selection_effect = sum(
            attr['selection_effect'] for attr in commodity_attribution.values()
        )

        total_allocation_effect = sum(
            attr['allocation_effect'] for attr in commodity_attribution.values()
        )

        return {
            'commodity_attribution': commodity_attribution,
            'total_selection_effect': total_selection_effect,
            'total_allocation_effect': total_allocation_effect,
            'total_attribution_effect': total_selection_effect + total_allocation_effect,
            'excess_return': excess_returns.mean(),
            'tracking_error': (portfolio_returns - benchmark_returns).std(),
            'information_ratio': excess_returns.mean() / (portfolio_returns - benchmark_returns).std() if (portfolio_returns - benchmark_returns).std() > 0 else 0,
            'benchmark_return': benchmark_returns.mean(),
            'portfolio_return': portfolio_returns.mean()
        }

    def optimize_dynamic_rebalancing(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        transaction_costs: Dict[str, float],
        rebalancing_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Optimize dynamic portfolio rebalancing strategy.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_costs: Transaction costs by commodity
            rebalancing_threshold: Minimum weight deviation for rebalancing

        Returns:
            Rebalancing strategy
        """
        rebalancing_trades = {}
        total_transaction_cost = 0

        for commodity in current_weights.keys():
            current_weight = current_weights[commodity]
            target_weight = target_weights.get(commodity, 0)

            weight_drift = abs(current_weight - target_weight)

            if weight_drift > rebalancing_threshold:
                # Calculate trade size
                if current_weight > target_weight:
                    # Sell commodity
                    trade_size = current_weight - target_weight
                    trade_direction = 'sell'
                else:
                    # Buy commodity
                    trade_size = target_weight - current_weight
                    trade_direction = 'buy'

                # Calculate transaction cost
                transaction_cost = trade_size * transaction_costs.get(commodity, 0.001)  # 0.1% default cost

                rebalancing_trades[commodity] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_drift': weight_drift,
                    'trade_direction': trade_direction,
                    'trade_size': trade_size,
                    'transaction_cost': transaction_cost,
                    'net_impact': -transaction_cost if trade_direction == 'sell' else transaction_cost
                }

                total_transaction_cost += transaction_cost

        # Calculate rebalancing efficiency
        total_weight_drift = sum(abs(current_weights[commodity] - target_weights.get(commodity, 0))
                                for commodity in current_weights.keys())

        rebalancing_efficiency = total_transaction_cost / total_weight_drift if total_weight_drift > 0 else 0

        return {
            'rebalancing_trades': rebalancing_trades,
            'total_transaction_cost': total_transaction_cost,
            'total_weight_drift': total_weight_drift,
            'rebalancing_efficiency': rebalancing_efficiency,
            'rebalancing_threshold': rebalancing_threshold,
            'trades_required': len(rebalancing_trades),
            'rebalancing_necessary': len(rebalancing_trades) > 0
        }

    def forecast_portfolio_risk(
        self,
        portfolio_weights: Dict[str, float],
        price_forecasts: Dict[str, pd.Series],
        correlation_forecast: Optional[pd.DataFrame] = None,
        forecast_horizon: int = 30
    ) -> pd.Series:
        """
        Forecast portfolio risk over time.

        Args:
            portfolio_weights: Portfolio weights by commodity
            price_forecasts: Price forecasts by commodity
            correlation_forecast: Forecasted correlation matrix
            forecast_horizon: Days to forecast

        Returns:
            Portfolio risk forecast
        """
        # Simple risk forecasting (could be enhanced with more sophisticated models)

        # Calculate expected portfolio volatility
        if correlation_forecast is not None:
            # Use forecasted correlations
            covariance_forecast = self._forecast_covariance_matrix(
                price_forecasts, correlation_forecast, portfolio_weights
            )
        else:
            # Use historical correlations (simplified)
            covariance_forecast = None

        # Generate risk forecast
        risk_forecast = []

        for day in range(forecast_horizon):
            if covariance_forecast is not None:
                # Use forecasted covariance
                weight_array = np.array([portfolio_weights.get(commodity, 0)
                                       for commodity in price_forecasts.keys()])
                daily_risk = np.sqrt(np.dot(weight_array.T, np.dot(covariance_forecast, weight_array)))
            else:
                # Use average historical risk (simplified)
                daily_risk = 0.15  # 15% daily volatility placeholder

            risk_forecast.append(daily_risk)

        # Create forecast series
        forecast_dates = pd.date_range(
            datetime.now() + pd.Timedelta(days=1),
            periods=forecast_horizon
        )

        return pd.Series(risk_forecast, index=forecast_dates)

    def _forecast_covariance_matrix(
        self,
        price_forecasts: Dict[str, pd.Series],
        correlation_forecast: pd.DataFrame,
        portfolio_weights: Dict[str, float]
    ) -> np.ndarray:
        """Forecast covariance matrix using price forecasts and correlations."""
        # Simplified covariance forecasting
        # In production: Use more sophisticated multivariate forecasting

        commodities = list(price_forecasts.keys())
        n = len(commodities)

        # Calculate forecasted volatilities
        volatilities = {}
        for commodity, forecast in price_forecasts.items():
            volatilities[commodity] = forecast.std()

        # Build covariance matrix
        covariance_matrix = np.zeros((n, n))

        for i, commodity1 in enumerate(commodities):
            for j, commodity2 in enumerate(commodities):
                if i == j:
                    # Diagonal: variance
                    covariance_matrix[i, j] = volatilities[commodity1] ** 2
                else:
                    # Off-diagonal: covariance = correlation * vol1 * vol2
                    correlation = correlation_forecast.loc[commodity1, commodity2]
                    covariance_matrix[i, j] = correlation * volatilities[commodity1] * volatilities[commodity2]

        return covariance_matrix

    def generate_portfolio_report(
        self,
        portfolio_weights: Dict[str, float],
        performance_data: Dict[str, Any],
        risk_metrics: Dict[str, float],
        attribution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio performance report.

        Args:
            portfolio_weights: Portfolio weights by commodity
            performance_data: Portfolio performance data
            risk_metrics: Portfolio risk metrics
            attribution_data: Performance attribution data

        Returns:
            Portfolio report
        """
        # Calculate portfolio statistics
        total_value = performance_data.get('total_portfolio_value', 1000000)  # $1M default

        # Generate recommendations
        recommendations = []

        # Risk-based recommendations
        if risk_metrics.get('portfolio_volatility', 0) > 0.25:  # 25% volatility threshold
            recommendations.append("Consider reducing exposure to high-volatility commodities")

        if risk_metrics.get('maximum_drawdown', 0) < -0.20:  # 20% drawdown threshold
            recommendations.append("Portfolio has experienced significant drawdowns. Consider risk management strategies.")

        # Diversification recommendations
        diversification_benefit = risk_metrics.get('diversification_benefit', 0)
        if diversification_benefit < 0.1:  # 10% diversification benefit threshold
            recommendations.append("Portfolio diversification benefits are limited. Consider adding uncorrelated assets.")

        # Rebalancing recommendations
        rebalancing_needed = performance_data.get('rebalancing_necessary', False)
        if rebalancing_needed:
            recommendations.append("Portfolio rebalancing recommended based on current weight drift.")

        return {
            'portfolio_summary': {
                'total_value': total_value,
                'commodities_held': len(portfolio_weights),
                'primary_commodities': list(portfolio_weights.keys())[:3],  # Top 3 by weight
                'portfolio_concentration': max(portfolio_weights.values())
            },
            'performance_metrics': performance_data,
            'risk_metrics': risk_metrics,
            'attribution_analysis': attribution_data,
            'recommendations': recommendations,
            'report_timestamp': datetime.now(),
            'report_period': performance_data.get('analysis_period', 'Last 30 days')
        }
