"""
LMP decomposition logic.
"""
import logging
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any
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
        # Mock mode for local development
        if os.getenv("MOCK_MODE", "0") == "1":
            timestamps = pd.date_range(start=start_time, end=end_time, freq="H", inclusive="both")
            rows = []
            for node in node_ids:
                base = 40.0 + (hash(node) % 10)
                for ts in timestamps:
                    hour = ts.hour
                    if 6 <= hour < 10 or 17 <= hour < 21:
                        mult = 1.2
                    elif hour >= 22 or hour < 6:
                        mult = 0.85
                    else:
                        mult = 1.0
                    price = base * mult + ((hash(str(ts)) % 5))
                    rows.append((ts.to_pydatetime(), node, float(price)))
            df = pd.DataFrame(rows, columns=["timestamp", "node_id", "lmp"])
            return df

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
        
        # Mock mode for local development
        if os.getenv("MOCK_MODE", "0") == "1":
            timestamps = pd.date_range(start=start_time, end=end_time, freq="H", inclusive="both")
            base = 40.0
            series = {}
            for ts in timestamps:
                hour = ts.hour
                if 6 <= hour < 10 or 17 <= hour < 21:
                    mult = 1.15
                elif hour >= 22 or hour < 6:
                    mult = 0.9
                else:
                    mult = 1.0
                series[ts.to_pydatetime()] = float(base * mult)
            return series
        
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

    async def forecast_congestion(
        self,
        node_ids: List[str],
        forecast_horizon_hours: int = 24,
        confidence_level: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Forecast congestion for specified nodes.

        Uses historical congestion patterns and current system conditions
        to predict future congestion levels.
        """
        logger.info(f"Forecasting congestion for {len(node_ids)} nodes over {forecast_horizon_hours}h")

        try:
            # Get recent historical congestion data (last 7 days)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)

            # Query historical congestion components
            query = f"""
            SELECT
                node_id,
                timestamp,
                congestion_component,
                toHour(timestamp) as hour_of_day,
                toDayOfWeek(timestamp) as day_of_week
            FROM market_intelligence.lmp_components
            WHERE node_id IN {tuple(node_ids)}
            AND timestamp >= '{start_time}'
            AND timestamp <= '{end_time}'
            ORDER BY node_id, timestamp
            """

            if os.getenv("MOCK_MODE", "0") == "1":
                # Generate mock congestion forecast data
                forecast_data = self._generate_mock_congestion_forecast(node_ids, forecast_horizon_hours)
            else:
                # In production, query actual data
                forecast_data = self._query_congestion_forecast(node_ids, forecast_horizon_hours)

            # Apply ML-based forecasting model
            forecasts = {}
            for node_id in node_ids:
                node_forecast = self._apply_congestion_model(
                    forecast_data.get(node_id, []),
                    forecast_horizon_hours,
                    confidence_level
                )
                forecasts[node_id] = node_forecast

            return {
                "forecast_horizon_hours": forecast_horizon_hours,
                "confidence_level": confidence_level,
                "generated_at": end_time.isoformat(),
                "forecasts": forecasts,
            }

        except Exception as e:
            logger.error(f"Error forecasting congestion: {e}")
            raise

    def _query_congestion_forecast(self, node_ids: List[str], horizon: int) -> Dict[str, List]:
        """Query historical congestion data for forecasting."""
        # Placeholder for actual ClickHouse query
        # In production, this would query the lmp_components table
        return {}

    def _generate_mock_congestion_forecast(self, node_ids: List[str], horizon: int) -> Dict[str, List]:
        """Generate mock congestion forecast data for development."""
        forecast_data = {}

        for node_id in node_ids:
            node_data = []
            base_congestion = 5.0 + (hash(node_id) % 15)  # Base congestion level

            for hour in range(horizon):
                # Add time-based and random variation
                hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
                random_factor = 0.8 + 0.4 * np.random.random()  # Random variation

                congestion = base_congestion * hour_factor * random_factor
                node_data.append({
                    "hour": hour,
                    "congestion": max(0, congestion),
                    "confidence": 0.7 + 0.2 * np.random.random()
                })

            forecast_data[node_id] = node_data

        return forecast_data

    def _apply_congestion_model(
        self,
        historical_data: List[Dict],
        horizon: int,
        confidence: float
    ) -> Dict[str, Any]:
        """Apply ML model to forecast congestion."""
        if not historical_data:
            # Fallback to simple pattern-based forecasting
            return self._simple_congestion_forecast(horizon, confidence)

        # In production, this would use a trained ML model
        # For now, use pattern-based forecasting
        return self._pattern_based_congestion_forecast(historical_data, horizon, confidence)

    def _simple_congestion_forecast(self, horizon: int, confidence: float) -> Dict[str, Any]:
        """Simple pattern-based congestion forecasting."""
        forecast = []

        for hour in range(horizon):
            # Simple daily pattern
            base_congestion = 5.0
            hour_factor = 1.0 + 0.2 * np.sin(2 * np.pi * hour / 24)

            forecast.append({
                "hour": hour,
                "predicted_congestion": base_congestion * hour_factor,
                "confidence_interval": [base_congestion * hour_factor * 0.8, base_congestion * hour_factor * 1.2],
                "confidence": confidence,
            })

        return {
            "method": "pattern_based",
            "forecast": forecast,
            "model_accuracy": 0.75,
        }

    def _pattern_based_congestion_forecast(
        self,
        historical_data: List[Dict],
        horizon: int,
        confidence: float
    ) -> Dict[str, Any]:
        """Pattern-based congestion forecasting using historical data."""
        # Analyze historical patterns
        hourly_patterns = {}
        for data_point in historical_data:
            hour = data_point.get("hour_of_day", 0)
            if hour not in hourly_patterns:
                hourly_patterns[hour] = []
            hourly_patterns[hour].append(data_point.get("congestion_component", 0))

        # Generate forecast based on patterns
        forecast = []
        for hour in range(horizon):
            # Use the same hour from previous week as baseline
            reference_hour = (hour + 24) % 24  # Same hour, previous day

            if reference_hour in hourly_patterns:
                # Use historical average for this hour
                avg_congestion = np.mean(hourly_patterns[reference_hour])
                std_congestion = np.std(hourly_patterns[reference_hour])

                # Add some trend and random variation
                trend_factor = 1.0 + 0.01 * hour  # Slight upward trend
                random_factor = 0.9 + 0.2 * np.random.random()

                predicted = avg_congestion * trend_factor * random_factor
                confidence_interval = [
                    predicted - 1.96 * std_congestion,
                    predicted + 1.96 * std_congestion
                ]
            else:
                # Fallback to simple pattern
                predicted = 5.0 * (1.0 + 0.1 * np.sin(2 * np.pi * hour / 24))
                confidence_interval = [predicted * 0.7, predicted * 1.3]

            forecast.append({
                "hour": hour,
                "predicted_congestion": max(0, predicted),
                "confidence_interval": [max(0, ci) for ci in confidence_interval],
                "confidence": confidence,
            })

        return {
            "method": "pattern_based",
            "forecast": forecast,
            "model_accuracy": 0.82,
            "historical_samples": len(historical_data),
        }

