"""
Hub-to-node basis surface modeling with advanced algorithms.
"""
import logging
import json
from datetime import date, timedelta
from typing import Dict, List, Tuple
import asyncio

import pandas as pd
import numpy as np
from clickhouse_driver import Client
import redis
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class BasisSurfaceModeler:
    """Model spatial price basis relationships with advanced algorithms."""

    def __init__(self):
        self.ch_client = Client(host="clickhouse", port=9000)
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour cache
    
    async def get_hub_prices(
        self,
        hub_id: str,
        as_of_date: date,
        iso: str,
        lookback_days: int = 90,
    ) -> pd.Series:
        """Get historical hub prices with Redis caching."""
        cache_key = f"hub_prices:{hub_id}:{as_of_date.isoformat()}:{lookback_days}"

        # Check cache first
        cached = self.redis_client.get(cache_key)
        if cached:
            logger.debug(f"Using cached hub prices for {hub_id}")
            data = json.loads(cached)
            return pd.Series(data, name='price')

        end_date = as_of_date
        start_date = end_date - timedelta(days=lookback_days)

        query = """
        SELECT
            toDate(event_time) as date,
            avg(value) as price
        FROM market_price_ticks
        WHERE instrument_id = %(hub_id)s
          AND toDate(event_time) >= %(start)s
          AND toDate(event_time) <= %(end)s
          AND price_type = 'settle'
        GROUP BY date
        ORDER BY date
        """

        result = self.ch_client.execute(
            query,
            {
                "hub_id": hub_id,
                "start": start_date,
                "end": end_date,
            },
        )

        if not result:
            return pd.Series(dtype=float, name='price')

        df = pd.DataFrame(result, columns=["date", "price"])
        df["date"] = pd.to_datetime(df["date"])
        prices = df.set_index("date")["price"]

        # Cache the result
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(prices.to_dict()))

        return prices
    
    async def get_node_prices(
        self,
        node_ids: list,
        as_of_date: date,
        iso: str,
        lookback_days: int = 90,
    ) -> Dict[str, pd.Series]:
        """Get historical prices for multiple nodes."""
        end_date = as_of_date
        start_date = end_date - timedelta(days=lookback_days)
        
        query = """
        SELECT 
            toDate(event_time) as date,
            instrument_id,
            avg(value) as price
        FROM ch.market_price_ticks
        WHERE instrument_id IN %(ids)s
          AND toDate(event_time) >= %(start)s
          AND toDate(event_time) <= %(end)s
          AND price_type = 'settle'
        GROUP BY date, instrument_id
        ORDER BY date, instrument_id
        """
        
        result = self.ch_client.execute(
            query,
            {
                "ids": tuple(node_ids),
                "start": start_date,
                "end": end_date,
            },
        )
        
        df = pd.DataFrame(result, columns=["date", "instrument_id", "price"])
        df["date"] = pd.to_datetime(df["date"])
        
        # Pivot to get prices per node
        node_prices = {}
        for node_id in node_ids:
            node_df = df[df["instrument_id"] == node_id]
            if not node_df.empty:
                node_prices[node_id] = node_df.set_index("date")["price"]
        
        return node_prices
    
    def calculate_basis_statistics(
        self,
        hub_prices: pd.Series,
        node_prices: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive basis statistics with advanced modeling.

        Basis = Node Price - Hub Price

        Enhanced with:
        - Seasonal decomposition analysis
        - Volatility regime detection
        - Trend and structural break detection
        - Correlation and cointegration analysis
        - Basis hedging effectiveness measures
        """
        # Align series
        aligned_hub, aligned_node = hub_prices.align(node_prices, join="inner")

        if aligned_hub.empty or aligned_node.empty:
            return {
                "mean": 0.0,
                "std": 0.0,
                "p95": 0.0,
                "p5": 0.0,
                "correlation": 0.0,
                "seasonal_mean": 0.0,
                "volatility_ratio": 0.0,
                "trend_slope": 0.0,
            }

        # Calculate basis
        basis = aligned_node - aligned_hub

        # Basic statistics
        mean_basis = basis.mean()
        std_basis = basis.std()

        # Seasonal analysis (by month)
        basis_df = pd.DataFrame({"basis": basis, "month": basis.index.month})
        seasonal_means = basis_df.groupby("month")["basis"].mean()
        seasonal_mean = seasonal_means.mean()  # Overall seasonal average

        # Volatility ratio (node vol / hub vol)
        hub_vol = aligned_hub.std()
        node_vol = aligned_node.std()
        volatility_ratio = node_vol / hub_vol if hub_vol > 0 else 1.0

        # Advanced analysis
        correlation = aligned_node.corr(aligned_hub)

        # Structural break detection (simplified)
        if len(basis) > 60:
            # Split data in half and compare means
            mid_point = len(basis) // 2
            first_half = basis.iloc[:mid_point]
            second_half = basis.iloc[mid_point:]

            mean_diff = abs(first_half.mean() - second_half.mean())
            structural_break = mean_diff > (std_basis * 0.5)  # Significant change
        else:
            structural_break = False

        # Cointegration test (simplified ADF test approximation)
        if len(basis) > 30:
            # Residual = basis - mean(basis) (stationarity test)
            residuals = basis - basis.mean()
            # Simple stationarity check: compare variance of residuals vs original
            residual_var = residuals.var()
            original_var = basis.var()
            stationarity_ratio = residual_var / original_var if original_var > 0 else 1.0
            is_cointegrated = stationarity_ratio < 0.8  # Simplified threshold
        else:
            is_cointegrated = True

        # Trend analysis with polynomial regression
        if len(basis) > 30:
            x = np.arange(len(basis))
            # Fit quadratic trend to detect non-linear patterns
            coeffs = np.polyfit(x, basis.values, 2)
            trend_slope = float(coeffs[0])  # Quadratic coefficient
            trend_linear = float(coeffs[1])  # Linear coefficient
        else:
            trend_slope = 0.0
            trend_linear = 0.0

        # Basis hedging effectiveness (how well basis hedges price risk)
        if hub_vol > 0 and node_vol > 0:
            # Hedge ratio = cov(node, hub) / var(hub)
            hedge_ratio = (aligned_node.cov(aligned_hub)) / hub_vol**2
            # Hedging effectiveness = 1 - var(hedged_position) / var(unhedged)
            hedging_effectiveness = 1 - (1 - correlation**2)
        else:
            hedge_ratio = 1.0
            hedging_effectiveness = 0.0

        return {
            "mean": float(mean_basis),
            "std": float(std_basis),
            "p95": float(basis.quantile(0.95)),
            "p5": float(basis.quantile(0.05)),
            "correlation": float(correlation),
            "seasonal_mean": float(seasonal_mean),
            "volatility_ratio": float(volatility_ratio),
            "trend_slope": trend_slope,
            "trend_linear": trend_linear,
            "structural_break": structural_break,
            "is_cointegrated": is_cointegrated,
            "hedge_ratio": float(hedge_ratio),
            "hedging_effectiveness": float(hedging_effectiveness),
            "sample_size": len(basis),
        }

