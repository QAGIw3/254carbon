"""
Hub-to-node basis surface modeling.
"""
import logging
from datetime import date, timedelta
from typing import Dict
import pandas as pd
import numpy as np
from clickhouse_driver import Client

logger = logging.getLogger(__name__)


class BasisSurfaceModeler:
    """Model spatial price basis relationships."""
    
    def __init__(self):
        self.ch_client = Client(host="clickhouse", port=9000)
    
    async def get_hub_prices(
        self,
        hub_id: str,
        as_of_date: date,
        iso: str,
        lookback_days: int = 90,
    ) -> pd.Series:
        """Get historical hub prices."""
        end_date = as_of_date
        start_date = end_date - timedelta(days=lookback_days)
        
        query = """
        SELECT 
            toDate(event_time) as date,
            avg(value) as price
        FROM ch.market_price_ticks
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
        
        df = pd.DataFrame(result, columns=["date", "price"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["price"]
    
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
        Calculate comprehensive basis statistics.

        Basis = Node Price - Hub Price

        Enhanced with:
        - Seasonal analysis
        - Volatility measures
        - Trend detection
        - Correlation analysis
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

        # Trend analysis (linear regression)
        if len(basis) > 30:  # Need sufficient data
            x = np.arange(len(basis))
            slope, _ = np.polyfit(x, basis.values, 1)
            trend_slope = float(slope)
        else:
            trend_slope = 0.0

        return {
            "mean": float(mean_basis),
            "std": float(std_basis),
            "p95": float(basis.quantile(0.95)),
            "p5": float(basis.quantile(0.05)),
            "correlation": float(aligned_node.corr(aligned_hub)),
            "seasonal_mean": float(seasonal_mean),
            "volatility_ratio": float(volatility_ratio),
            "trend_slope": trend_slope,
        }

