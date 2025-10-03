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
        distance_from_hub: float = None,
    ) -> float:
        """
        Calculate marginal loss component using distance-based estimation.

        Loss = Energy * Loss_Factor * Distance_Factor

        In production, would use actual loss sensitivity factors from ISO.
        """
        loss_factor = self.loss_factors.get(iso, 0.015)

        # Distance factor: farther nodes have higher losses
        if distance_from_hub is None:
            # Estimate based on node ID (simplified heuristic)
            distance_factor = 1.0 + (hash(node_id) % 100) / 1000.0  # 0-10% variation
        else:
            # Use actual electrical distance
            distance_factor = 1.0 + distance_from_hub / 100.0  # Scale by distance

        # Loss component is percentage of energy price
        loss = energy_price * loss_factor * distance_factor

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

    async def calculate_electrical_distance(
        self,
        node_id: str,
        iso: str,
        network: Dict[str, Any],
    ) -> float:
        """
        Calculate electrical distance from reference hub to node.

        Uses shortest path in network graph weighted by reactance.
        """
        try:
            import networkx as nx

            # Build network graph
            G = nx.Graph()
            for line in network["lines"]:
                G.add_edge(
                    line["from"],
                    line["to"],
                    weight=line["reactance"]  # Use reactance as edge weight
                )

            ref_bus = network["reference_bus"]

            if node_id not in G.nodes() or ref_bus not in G.nodes():
                return 10.0  # Default distance

            # Calculate shortest path distance
            try:
                distance = nx.shortest_path_length(G, ref_bus, node_id, weight="weight")
                return float(distance)
            except nx.NetworkXNoPath:
                return 20.0  # Large distance if no path

        except Exception as e:
            logger.error(f"Error calculating electrical distance: {e}")
            return 10.0  # Default fallback

    async def calculate_congestion_component(
        self,
        node_id: str,
        energy: float,
        loss: float,
        lmp_total: float,
        timestamp: datetime,
        iso: str,
    ) -> float:
        """
        Calculate congestion component with constraint awareness.

        In production, would use shadow prices from OPF solution.
        For now, estimate based on historical patterns and binding constraints.
        """
        # Get historical congestion for this node
        historical_congestion = await self.get_historical_congestion(
            node_id, lookback_days=30, iso=iso
        )

        if not historical_congestion.empty:
            # Use recent average as baseline
            baseline_congestion = historical_congestion.mean()

            # Identify binding constraints for this timestamp
            binding_constraints = await self.identify_binding_constraints(node_id, iso)

            # Adjust for constraint impact (simplified)
            constraint_multiplier = 1.0
            for constraint in binding_constraints[:3]:  # Top 3 constraints
                constraint_multiplier *= (1.0 + constraint["binding_frequency"] * 0.5)

            congestion = baseline_congestion * constraint_multiplier
        else:
            # Fallback: estimate as residual after energy and loss
            congestion = max(0, lmp_total - energy - loss)

        return congestion

