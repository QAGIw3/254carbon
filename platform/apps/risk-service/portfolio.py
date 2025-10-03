"""
Portfolio aggregation and analytics.
"""
import logging
from typing import List, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioAggregator:
    """Aggregate positions and calculate portfolio-level metrics."""
    
    def aggregate_positions(
        self,
        positions: List,
        prices: pd.DataFrame,
    ) -> Dict:
        """
        Aggregate positions into portfolio summary.
        
        Returns portfolio composition, exposures, concentrations.
        """
        # Calculate position values
        position_values = []
        
        for pos in positions:
            if pos.instrument_id in prices.columns:
                current_price = prices[pos.instrument_id].iloc[-1]
                value = pos.quantity * current_price
                
                position_values.append({
                    "instrument_id": pos.instrument_id,
                    "quantity": pos.quantity,
                    "current_price": current_price,
                    "value": value,
                    "pnl": value - (pos.quantity * pos.entry_price)
                    if pos.entry_price else 0,
                })
        
        # Total portfolio value
        total_value = sum(p["value"] for p in position_values)
        
        # Calculate concentrations
        concentrations = [
            {
                "instrument_id": p["instrument_id"],
                "weight": p["value"] / total_value if total_value > 0 else 0,
            }
            for p in position_values
        ]
        
        # Sort by weight
        concentrations.sort(key=lambda x: x["weight"], reverse=True)
        
        return {
            "total_value": total_value,
            "positions": position_values,
            "concentrations": concentrations,
            "largest_position_pct": concentrations[0]["weight"] * 100
            if concentrations else 0,
        }

