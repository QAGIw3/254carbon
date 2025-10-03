"""
LMP decomposition logic.
"""
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List
import numpy as np
import pandas as pd
from clickhouse_driver import Client

logger = logging.getLogger(__name__)


class LMPDecomposer:
    """Decompose LMP into components."""
    
    def __init__(self):
        self.ch_client = Client(host="clickhouse", port=9000)
        
        # Loss factors by ISO (typical values)
        self.loss_factors = {
            "PJM": 0.015,  # 1.5%
            "MISO": 0.012,  # 1.2%
            "ERCOT": 0.020,  # 2.0%
            "CAISO": 0.018,  # 1.8%
        }
        
        # Reference hubs by ISO
        self.reference_hubs = {
            "PJM": "PJM.HUB.WEST",
            "MISO": "MISO.HUB.INDIANA",
            "ERCOT": "ERCOT.HUB.NORTH",
            "CAISO": "CAISO.HUB.SP15",
        }
    
    async def get_lmp_data(
        self,
        node_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        iso: str,
    ) -> pd.DataFrame:
        """Get raw LMP data from ClickHouse."""
        query = """
        SELECT 
            event_time as timestamp,
            instrument_id as node_id,
            value as lmp
        FROM ch.market_price_ticks
        WHERE instrument_id IN %(ids)s
          AND event_time BETWEEN %(start)s AND %(end)s
          AND price_type IN ('trade', 'settle')
        ORDER BY event_time, instrument_id
        """
        
        result = self.ch_client.execute(
            query,
            {
                "ids": tuple(node_ids),
                "start": start_time,
                "end": end_time,
            },
        )
        
        return pd.DataFrame(result, columns=["timestamp", "node_id", "lmp"])
    
    async def get_energy_component(
        self,
        iso: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[datetime, float]:
        """Get energy component (hub price) time series."""
        hub_id = self.reference_hubs.get(iso)
        
        if not hub_id:
            logger.warning(f"No reference hub for {iso}, using fallback")
            return {}
        
        query = """
        SELECT 
            event_time as timestamp,
            value as energy_price
        FROM ch.market_price_ticks
        WHERE instrument_id = %(hub_id)s
          AND event_time BETWEEN %(start)s AND %(end)s
          AND price_type IN ('trade', 'settle')
        ORDER BY event_time
        """
        
        result = self.ch_client.execute(
            query,
            {
                "hub_id": hub_id,
                "start": start_time,
                "end": end_time,
            },
        )
        
        return {row[0]: row[1] for row in result}
    
    def calculate_loss_component(
        self,
        node_id: str,
        energy_price: float,
        iso: str,
    ) -> float:
        """
        Calculate marginal loss component.
        
        Loss = Energy * Loss_Factor
        
        In practice, would use actual loss sensitivity factors.
        """
        loss_factor = self.loss_factors.get(iso, 0.015)
        
        # Simplified: loss component is a percentage of energy
        loss = energy_price * loss_factor
        
        return loss
    
    async def get_historical_congestion(
        self,
        node_id: str,
        lookback_days: int,
        iso: str,
    ) -> pd.Series:
        """Get historical congestion component."""
        # In production, would query pre-computed decomposition
        # For now, estimate from price spreads
        
        hub_id = self.reference_hubs.get(iso)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get node vs hub prices
        query = """
        SELECT 
            toDate(event_time) as date,
            instrument_id,
            avg(value) as avg_price
        FROM ch.market_price_ticks
        WHERE instrument_id IN (%(node_id)s, %(hub_id)s)
          AND event_time >= %(start)s
          AND price_type = 'settle'
        GROUP BY date, instrument_id
        ORDER BY date
        """
        
        result = self.ch_client.execute(
            query,
            {
                "node_id": node_id,
                "hub_id": hub_id,
                "start": start_date,
            },
        )
        
        df = pd.DataFrame(result, columns=["date", "instrument_id", "avg_price"])
        pivot = df.pivot(index="date", columns="instrument_id", values="avg_price")
        
        if node_id in pivot.columns and hub_id in pivot.columns:
            # Congestion ≈ Node price - Hub price - Loss
            loss = pivot[hub_id] * self.loss_factors.get(iso, 0.015)
            congestion = pivot[node_id] - pivot[hub_id] - loss
            return congestion
        
        return pd.Series()
    
    async def identify_binding_constraints(
        self,
        node_id: str,
        iso: str,
    ) -> List[Dict[str, Any]]:
        """
        Identify constraints that frequently bind at this node.
        
        In production, would query constraint shadow prices.
        """
        # Mock data - in production would query ISO constraint data
        mock_constraints = [
            {"constraint_id": f"{iso}_LINE_101", "binding_frequency": 0.45},
            {"constraint_id": f"{iso}_LINE_202", "binding_frequency": 0.32},
            {"constraint_id": f"{iso}_INTERFACE_A", "binding_frequency": 0.28},
        ]
        
        return sorted(
            mock_constraints,
            key=lambda x: x["binding_frequency"],
            reverse=True,
        )
    
    def forecast_nodal_congestion(
        self,
        node_id: str,
        forecast_date: date,
        historical_congestion: pd.Series,
        binding_constraints: List[Dict],
    ) -> float:
        """
        Forecast congestion component.
        
        Simple approach: historical average with seasonal adjustment.
        """
        if historical_congestion.empty:
            return 0.0
        
        # Calculate statistics
        mean_congestion = historical_congestion.mean()
        std_congestion = historical_congestion.std()
        
        # Seasonal factor (simplified)
        month = forecast_date.month
        summer_months = [6, 7, 8, 9]
        seasonal_factor = 1.3 if month in summer_months else 0.9
        
        # Forecast
        forecast = mean_congestion * seasonal_factor
        
        return float(forecast)

