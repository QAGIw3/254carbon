"""
Advanced Analytics API Endpoints

Endpoints for advanced analytics and research:
- Correlation matrix calculations
- Volatility surface data
- Seasonality decomposition
- Custom research queries
- Portfolio optimization results
"""
import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Body
from pydantic import BaseModel
import pandas as pd
import numpy as np

from auth import verify_token, has_permission
from entitlements import check_entitlement
from db import get_clickhouse_client
from cache import create_cache_decorator, CacheStrategy

logger = logging.getLogger(__name__)

# Create router
analytics_router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["analytics"],
    dependencies=[Depends(verify_token)]
)


class CorrelationMatrixResponse(BaseModel):
    commodities: List[str]
    correlation_matrix: List[List[float]]
    timestamp: datetime
    lookback_days: int


class VolatilitySurfaceResponse(BaseModel):
    commodity: str
    volatility_surface: Dict[str, List[float]]
    timestamp: datetime
    horizons: List[int]


class SeasonalityResponse(BaseModel):
    commodity: str
    seasonal_pattern: Dict[str, float]
    timestamp: datetime
    analysis_period: str


class PortfolioOptimizationResponse(BaseModel):
    portfolio_weights: Dict[str, float]
    expected_return: float
    portfolio_risk: float
    sharpe_ratio: float
    optimization_method: str


@analytics_router.get("/correlations/matrix", response_model=CorrelationMatrixResponse)
@create_cache_decorator("correlation_matrix", CacheStrategy.STATIC)
async def get_correlation_matrix(
    commodities: List[str] = Query(..., description="Commodities to include in correlation analysis"),
    lookback_days: int = Query(default=90, description="Lookback period in days"),
    start_date: Optional[date] = Query(None, description="Start date for analysis")
):
    """Get correlation matrix for multiple commodities."""
    try:
        # Check entitlements for analytics access
        await check_entitlement("analytics_access")

        ch = get_clickhouse_client()

        # Determine date range
        if start_date:
            start_datetime = datetime.combine(start_date, datetime.min.time())
        else:
            start_datetime = datetime.now() - timedelta(days=lookback_days)

        end_datetime = datetime.now()

        # Query price data for correlation calculation
        query = """
        SELECT
            instrument_id,
            event_time,
            value
        FROM market_intelligence.market_price_ticks
        WHERE instrument_id IN %(commodities)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY instrument_id, event_time
        """

        result = ch.execute(
            query,
            {
                'commodities': tuple(commodities),
                'start_date': start_datetime,
                'end_date': end_datetime
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No price data found for correlation analysis")

        # Convert to DataFrame for correlation calculation
        df = pd.DataFrame(result, columns=['instrument_id', 'event_time', 'value'])

        # Pivot to get price matrix
        price_matrix = df.pivot(index='event_time', columns='instrument_id', values='value')

        # Calculate correlation matrix
        correlation_matrix = price_matrix.corr()

        # Convert to list format for JSON response
        corr_list = correlation_matrix.values.tolist()

        return CorrelationMatrixResponse(
            commodities=commodities,
            correlation_matrix=corr_list,
            timestamp=datetime.now(),
            lookback_days=lookback_days
        )

    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/volatility/surface/{commodity}", response_model=VolatilitySurfaceResponse)
@create_cache_decorator("vol_surface", CacheStrategy.SEMI_STATIC)
async def get_volatility_surface(
    commodity: str,
    horizons: List[int] = Query(default=[30, 90, 180, 365], description="Time horizons for volatility"),
    start_date: Optional[date] = Query(None, description="Start date for analysis")
):
    """Get volatility surface for a commodity across different horizons."""
    try:
        # Check entitlements for analytics access
        await check_entitlement("analytics_access")

        ch = get_clickhouse_client()

        # Determine date range
        if start_date:
            start_datetime = datetime.combine(start_date, datetime.min.time())
        else:
            start_datetime = datetime.now() - timedelta(days=max(horizons))

        end_datetime = datetime.now()

        # Query price data for volatility calculation
        query = """
        SELECT
            instrument_id,
            event_time,
            value
        FROM market_intelligence.market_price_ticks
        WHERE instrument_id = %(commodity)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time
        """

        result = ch.execute(
            query,
            {
                'commodity': commodity,
                'start_date': start_datetime,
                'end_date': end_datetime
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"No price data found for {commodity}")

        # Convert to DataFrame
        df = pd.DataFrame(result, columns=['instrument_id', 'event_time', 'value'])

        # Calculate volatility for different horizons
        volatility_surface = {}

        for horizon in horizons:
            # Calculate rolling volatility
            returns = df['value'].pct_change()
            rolling_vol = returns.rolling(horizon).std() * np.sqrt(252)  # Annualized

            # Get volatility values for different periods
            volatility_values = rolling_vol.dropna().tolist()
            volatility_surface[f'{horizon}_days'] = volatility_values

        return VolatilitySurfaceResponse(
            commodity=commodity,
            volatility_surface=volatility_surface,
            timestamp=datetime.now(),
            horizons=horizons
        )

    except Exception as e:
        logger.error(f"Error calculating volatility surface: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/seasonality/{commodity}", response_model=SeasonalityResponse)
@create_cache_decorator("seasonality", CacheStrategy.STATIC)
async def get_seasonality_analysis(
    commodity: str,
    analysis_period: str = Query(default="3_years", description="Analysis period"),
    start_date: Optional[date] = Query(None, description="Start date for analysis")
):
    """Get seasonality analysis for a commodity."""
    try:
        # Check entitlements for analytics access
        await check_entitlement("analytics_access")

        ch = get_clickhouse_client()

        # Determine analysis period
        if analysis_period == "1_year":
            days_back = 365
        elif analysis_period == "3_years":
            days_back = 1095
        elif analysis_period == "5_years":
            days_back = 1825
        else:
            days_back = 1095  # Default to 3 years

        # Determine date range
        if start_date:
            start_datetime = datetime.combine(start_date, datetime.min.time())
        else:
            start_datetime = datetime.now() - timedelta(days=days_back)

        end_datetime = datetime.now()

        # Query price data for seasonality analysis
        query = """
        SELECT
            instrument_id,
            event_time,
            value
        FROM market_intelligence.market_price_ticks
        WHERE instrument_id = %(commodity)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time
        """

        result = ch.execute(
            query,
            {
                'commodity': commodity,
                'start_date': start_datetime,
                'end_date': end_datetime
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"No price data found for {commodity}")

        # Convert to DataFrame
        df = pd.DataFrame(result, columns=['instrument_id', 'event_time', 'value'])

        # Calculate seasonal patterns
        seasonal_pattern = {}

        # Monthly seasonality
        monthly_avg = df.groupby(df['event_time'].dt.month)['value'].mean()
        for month, avg_price in monthly_avg.items():
            seasonal_pattern[f'month_{month}'] = avg_price

        # Daily seasonality
        daily_avg = df.groupby(df['event_time'].dt.dayofweek)['value'].mean()
        for day, avg_price in daily_avg.items():
            seasonal_pattern[f'day_{day}'] = avg_price

        # Hourly seasonality (if data is frequent enough)
        if len(df) > 1000:  # Sufficient data for hourly analysis
            hourly_avg = df.groupby(df['event_time'].dt.hour)['value'].mean()
            for hour, avg_price in hourly_avg.items():
                seasonal_pattern[f'hour_{hour}'] = avg_price

        return SeasonalityResponse(
            commodity=commodity,
            seasonal_pattern=seasonal_pattern,
            timestamp=datetime.now(),
            analysis_period=f"{start_datetime.date()} to {end_datetime.date()}"
        )

    except Exception as e:
        logger.error(f"Error calculating seasonality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class OptimizePortfolioRequest(BaseModel):
    commodities: List[str]
    target_return: Optional[float] = None
    risk_tolerance: str = "moderate"


@analytics_router.post("/portfolio/optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(
    request: OptimizePortfolioRequest
):
    """Optimize multi-commodity portfolio allocation."""
    try:
        # Check entitlements for portfolio optimization
        await check_entitlement("portfolio_optimization")

        # Import portfolio optimizer (would be implemented)
        from multi_commodity_portfolio_optimizer import MultiCommodityPortfolioOptimizer

        optimizer = MultiCommodityPortfolioOptimizer()

        # Get historical returns data (simplified for demo)
        # In production: Query actual returns data
        returns_data = pd.DataFrame()  # Placeholder

        # Set risk tolerance parameters
        risk_tolerance = request.risk_tolerance
        if risk_tolerance == "conservative":
            max_weight = 0.15
        elif risk_tolerance == "moderate":
            max_weight = 0.25
        elif risk_tolerance == "aggressive":
            max_weight = 0.35
        else:
            max_weight = 0.25

        # Run optimization
        optimization_result = optimizer.optimize_portfolio_weights(
            returns_data=returns_data,
            target_return=request.target_return,
            max_weight=max_weight,
            optimization_method='mean_variance'
        )

        return PortfolioOptimizationResponse(
            portfolio_weights=optimization_result['optimal_weights'],
            expected_return=optimization_result['portfolio_metrics']['expected_return'],
            portfolio_risk=optimization_result['portfolio_metrics']['portfolio_risk'],
            sharpe_ratio=optimization_result['portfolio_metrics']['sharpe_ratio'],
            optimization_method=optimization_result['optimization_method']
        )

    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/arbitrage/opportunities")
async def detect_arbitrage_opportunities(
    commodities: List[str] = Query(..., description="Commodities to analyze"),
    min_profit_threshold: float = Query(default=0.05, description="Minimum profit threshold")
):
    """Detect cross-commodity arbitrage opportunities."""
    try:
        # Check entitlements for arbitrage analysis
        await check_entitlement("arbitrage_analysis")

        # Import arbitrage detector (would be implemented)
        from multi_commodity_portfolio_optimizer import MultiCommodityPortfolioOptimizer

        optimizer = MultiCommodityPortfolioOptimizer()

        # Get price data for arbitrage analysis (simplified)
        price_data = {}  # Placeholder

        # Detect arbitrage opportunities
        arbitrage_result = optimizer.detect_cross_market_arbitrage(
            price_data=price_data,
            arbitrage_threshold=min_profit_threshold
        )

        return arbitrage_result

    except Exception as e:
        logger.error(f"Error detecting arbitrage opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class StressTestRequest(BaseModel):
    portfolio_weights: Dict[str, float]
    stress_scenarios: List[str] = ["oil_shock", "carbon_price_spike"]


@analytics_router.post("/stress-test/portfolio")
async def stress_test_portfolio(
    request: StressTestRequest
):
    """Perform stress testing on portfolio under various scenarios."""
    try:
        # Check entitlements for stress testing
        await check_entitlement("stress_testing")

        # Import portfolio optimizer for stress testing
        from multi_commodity_portfolio_optimizer import MultiCommodityPortfolioOptimizer

        optimizer = MultiCommodityPortfolioOptimizer()

        # Define stress scenarios
        scenario_definitions = []

        for scenario_name in stress_scenarios:
            if scenario_name == "oil_shock":
                scenario_definitions.append({
                    'name': 'oil_shock',
                    'description': 'Oil price shock scenario',
                    'shocks': [
                        {'commodity': 'WTI', 'type': 'relative', 'magnitude': 0.5},  # 50% price increase
                        {'commodity': 'BRENT', 'type': 'relative', 'magnitude': 0.4}
                    ]
                })
            elif scenario_name == "carbon_price_spike":
                scenario_definitions.append({
                    'name': 'carbon_price_spike',
                    'description': 'Carbon price spike scenario',
                    'shocks': [
                        {'commodity': 'EUA', 'type': 'relative', 'magnitude': 1.0},  # Double carbon prices
                        {'commodity': 'CCA', 'type': 'relative', 'magnitude': 0.8}
                    ]
                })

        # Get base returns data (simplified)
        base_returns = pd.DataFrame()  # Placeholder

        # Run stress testing
        stress_results = optimizer.perform_scenario_stress_testing(
            portfolio_weights=request.portfolio_weights,
            scenario_definitions=scenario_definitions,
            base_returns=base_returns
        )

        return stress_results

    except Exception as e:
        logger.error(f"Error performing stress testing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PerformanceAttributionRequest(BaseModel):
    portfolio_returns: List[float]
    benchmark_returns: List[float]
    commodity_returns: Dict[str, List[float]]
    portfolio_weights: Dict[str, float]


@analytics_router.post("/performance/attribution")
async def get_performance_attribution(
    request: PerformanceAttributionRequest
):
    """Calculate performance attribution for portfolio."""
    try:
        # Check entitlements for performance analysis
        await check_entitlement("performance_attribution")

        # Import portfolio optimizer for attribution
        from multi_commodity_portfolio_optimizer import MultiCommodityPortfolioOptimizer

        optimizer = MultiCommodityPortfolioOptimizer()

        # Convert lists to pandas Series
        portfolio_series = pd.Series(request.portfolio_returns)
        benchmark_series = pd.Series(request.benchmark_returns)

        # Convert commodity returns dict to DataFrame
        commodity_df = pd.DataFrame(request.commodity_returns)

        # Run attribution analysis
        attribution_result = optimizer.calculate_performance_attribution(
            portfolio_returns=portfolio_series,
            benchmark_returns=benchmark_series,
            commodity_returns=commodity_df,
            portfolio_weights=request.portfolio_weights
        )

        return attribution_result

    except Exception as e:
        logger.error(f"Error calculating performance attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CustomQueryRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None


@analytics_router.post("/custom-query")
async def execute_custom_query(
    request: CustomQueryRequest
):
    """Execute custom research queries on ClickHouse data."""
    try:
        # Check entitlements for custom query access
        await check_entitlement("custom_queries")

        ch = get_clickhouse_client()

        # Execute custom query
        if request.parameters:
            result = ch.execute(request.query, request.parameters)
        else:
            result = ch.execute(request.query)

        return {
            'query': request.query,
            'parameters': request.parameters,
            'result_count': len(result),
            'results': result[:1000],  # Limit results for API response
            'timestamp': datetime.now()
        }

    except Exception as e:
        logger.error(f"Error executing custom query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/data-quality/report")
async def get_data_quality_report(
    sources: List[str] = Query(default=[], description="Data sources to analyze"),
    commodity_types: List[str] = Query(default=[], description="Commodity types to analyze")
):
    """Get comprehensive data quality report."""
    try:
        # Check entitlements for data quality access
        await check_entitlement("data_quality_analysis")

        # Import data quality framework
        from data_quality_framework import DataQualityFramework

        quality_framework = DataQualityFramework()

        # Get data sources for analysis (simplified)
        data_sources = {}  # Placeholder - would query actual data

        # Generate quality report
        quality_report = quality_framework.generate_quality_report(
            data_sources=data_sources,
            commodity_types={source: 'unknown' for source in sources}  # Placeholder
        )

        return quality_report

    except Exception as e:
        logger.error(f"Error generating data quality report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/dq/issues")
async def list_dq_issues(
    instrument_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100)
):
    """List data quality issues from ClickHouse."""
    try:
        await check_entitlement("data_quality_analysis")
        ch = await get_clickhouse_client()
        clauses = ["1=1"]
        params: Dict[str, Any] = {}
        if instrument_id:
            clauses.append("instrument_id = %(instrument_id)s")
            params["instrument_id"] = instrument_id
        if severity:
            clauses.append("severity = %(severity)s")
            params["severity"] = severity
        if start_time:
            clauses.append("event_time >= %(start_time)s")
            params["start_time"] = start_time
        if end_time:
            clauses.append("event_time <= %(end_time)s")
            params["end_time"] = end_time
        where_sql = " AND ".join(clauses)
        sql = f"""
        SELECT event_time, source, instrument_id, commodity_type, dimension, severity, rule_id, value, expected, metadata
        FROM market_intelligence.data_quality_issues
        WHERE {where_sql}
        ORDER BY event_time DESC
        LIMIT %(limit)s
        """
        params["limit"] = limit
        rows = await ch.fetch(sql, parameters=params)
        return {"count": len(rows), "issues": rows}
    except Exception as e:
        logger.error(f"Error listing DQ issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/dq/scores")
async def list_dq_scores(
    source: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None)
):
    try:
        await check_entitlement("data_quality_analysis")
        ch = await get_clickhouse_client()
        clauses = ["1=1"]
        params: Dict[str, Any] = {}
        if source:
            clauses.append("source = %(source)s")
            params["source"] = source
        if start_date:
            clauses.append("date >= %(start_date)s")
            params["start_date"] = start_date
        if end_date:
            clauses.append("date <= %(end_date)s")
            params["end_date"] = end_date
        where_sql = " AND ".join(clauses)
        sql = f"""
        SELECT date, source, commodity_type, score, components
        FROM market_intelligence.data_quality_scores
        WHERE {where_sql}
        ORDER BY date DESC
        LIMIT 1000
        """
        rows = await ch.fetch(sql, parameters=params)
        return {"count": len(rows), "scores": rows}
    except Exception as e:
        logger.error(f"Error listing DQ scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/dq/validation")
async def list_cross_source_validation(
    instrument_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(200)
):
    try:
        await check_entitlement("data_quality_analysis")
        ch = await get_clickhouse_client()
        clauses = ["1=1"]
        params: Dict[str, Any] = {}
        if instrument_id:
            clauses.append("instrument_id = %(instrument_id)s")
            params["instrument_id"] = instrument_id
        if start_time:
            clauses.append("ts >= %(start_time)s")
            params["start_time"] = start_time
        if end_time:
            clauses.append("ts <= %(end_time)s")
            params["end_time"] = end_time
        where_sql = " AND ".join(clauses)
        sql = f"""
        SELECT ts, primary_source, secondary_source, instrument_id, metric_name, rel_diff, within_tolerance, reconciled_value
        FROM market_intelligence.cross_source_validation
        WHERE {where_sql}
        ORDER BY ts DESC
        LIMIT %(limit)s
        """
        params["limit"] = limit
        rows = await ch.fetch(sql, parameters=params)
        return {"count": len(rows), "validation": rows}
    except Exception as e:
        logger.error(f"Error listing cross-source validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/lineage/trace/{entry_id}")
async def trace_data_lineage(
    entry_id: str,
    max_depth: int = Query(default=10, description="Maximum lineage depth")
):
    """Trace data lineage for audit purposes."""
    try:
        # Check entitlements for lineage access
        await check_entitlement("data_lineage")

        # Import lineage tracker
        from data_quality_framework import DataLineageTracker

        lineage_tracker = DataLineageTracker()

        # Get lineage chain
        lineage_chain = lineage_tracker.get_lineage_chain(entry_id)

        return {
            'entry_id': entry_id,
            'lineage_chain': lineage_chain,
            'max_depth': max_depth,
            'chain_length': len(lineage_chain),
            'timestamp': datetime.now()
        }

    except Exception as e:
        logger.error(f"Error tracing data lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))
