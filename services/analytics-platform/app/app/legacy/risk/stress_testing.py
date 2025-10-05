"""
Stress testing engine for portfolio analysis.
"""
import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StressTestEngine:
    """Apply stress scenarios to portfolios."""
    
    async def apply_scenario(
        self,
        positions: List,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply stress scenario to portfolio.
        
        Scenario types:
        - price_shock: Parallel shift in all prices
        - volatility_spike: Increase in volatility
        - correlation_breakdown: Correlations go to 1 or -1
        - historical_event: Replay specific crisis
        """
        scenario_type = scenario.get("type", "price_shock")
        
        if scenario_type == "price_shock":
            return await self._price_shock(positions, scenario)
        elif scenario_type == "volatility_spike":
            return await self._volatility_spike(positions, scenario)
        elif scenario_type == "correlation_breakdown":
            return await self._correlation_breakdown(positions, scenario)
        elif scenario_type == "historical_event":
            return await self._historical_event(positions, scenario)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    async def _price_shock(
        self,
        positions: List,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply price shock scenario.
        
        Example: {"type": "price_shock", "shock_pct": -20}
        """
        shock_pct = scenario.get("shock_pct", -20) / 100
        
        # Calculate impact on each position
        total_pnl = 0.0
        position_impacts = []
        
        for pos in positions:
            # Simple shock: P&L = quantity * price * shock%
            # In production, would get actual current prices
            base_value = pos.quantity * (pos.entry_price or 50.0)
            shocked_value = base_value * (1 + shock_pct)
            pnl = shocked_value - base_value
            
            total_pnl += pnl
            
            position_impacts.append({
                "instrument_id": pos.instrument_id,
                "base_value": base_value,
                "shocked_value": shocked_value,
                "pnl": pnl,
            })
        
        return {
            "pnl": total_pnl,
            "pnl_pct": shock_pct * 100,
            "details": {
                "shock_pct": shock_pct * 100,
                "positions": position_impacts,
            },
        }
    
    async def _volatility_spike(
        self,
        positions: List,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply volatility spike scenario.
        
        Higher volatility increases option values and risk metrics.
        """
        vol_multiplier = scenario.get("vol_multiplier", 2.0)
        
        # Simplified: assume portfolio value changes with vol
        # In practice, would recalculate option Greeks
        base_volatility = 0.30  # 30% annualized
        new_volatility = base_volatility * vol_multiplier
        
        # Impact depends on portfolio composition
        # For simplicity, assume 10% value impact per doubling of vol
        impact_pct = (vol_multiplier - 1) * 0.10
        
        total_value = sum(
            pos.quantity * (pos.entry_price or 50.0)
            for pos in positions
        )
        
        pnl = total_value * impact_pct
        
        return {
            "pnl": pnl,
            "pnl_pct": impact_pct * 100,
            "details": {
                "base_volatility": base_volatility,
                "stressed_volatility": new_volatility,
                "vol_multiplier": vol_multiplier,
            },
        }
    
    async def _correlation_breakdown(
        self,
        positions: List,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Correlation breakdown scenario.
        
        Diversification benefits disappear.
        """
        # Assume all correlations go to 1 (perfect correlation)
        # Portfolio volatility = sum of individual volatilities
        
        return {
            "pnl": 0.0,  # No immediate P&L impact
            "pnl_pct": 0.0,
            "details": {
                "message": "Diversification benefits lost",
                "note": "VaR would increase significantly",
            },
        }
    
    async def _historical_event(
        self,
        positions: List,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Replay historical crisis event.
        
        Example: 2008 financial crisis, 2021 Texas freeze
        """
        event_name = scenario.get("event", "2008_financial_crisis")
        
        # Predefined historical shocks
        event_shocks = {
            "2008_financial_crisis": -0.40,  # -40%
            "2021_texas_freeze": 0.50,  # +50% for power prices
            "2000_california_crisis": 1.00,  # +100%
        }
        
        shock = event_shocks.get(event_name, -0.20)
        
        # Apply shock
        return await self._price_shock(
            positions,
            {"type": "price_shock", "shock_pct": shock * 100},
        )

