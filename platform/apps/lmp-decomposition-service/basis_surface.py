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
        Calculate basis statistics.
        
        Basis = Node Price - Hub Price
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
            }
        
        # Calculate basis
        basis = aligned_node - aligned_hub
        
        return {
            "mean": float(basis.mean()),
            "std": float(basis.std()),
            "p95": float(basis.quantile(0.95)),
            "p5": float(basis.quantile(0.05)),
            "correlation": float(aligned_node.corr(aligned_hub)),
        }

