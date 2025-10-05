"""
Minimal fallback for CommodityFeatureEngineer.
Provides a basic implementation of generation spread feature computation
so that services can start without the full dependency tree.
"""
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd


class CommodityFeatureEngineer:
    def build_generation_spread_features(
        self,
        *,
        power_prices: pd.Series,
        gas_prices: Optional[pd.Series] = None,
        coal_prices: Optional[pd.Series] = None,
        carbon_prices: Optional[pd.Series] = None,
        heat_rate_gas: Optional[float] = None,
        heat_rate_coal: Optional[float] = None,
        emissions_factor_gas: Optional[float] = None,
        emissions_factor_coal: Optional[float] = None,
        fallback_capacity: Optional[float] = None,
        capacity_series: Optional[pd.Series] = None,
        load_series: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        spark_spread = 0.0
        dark_spread = 0.0

        if power_prices is not None and not power_prices.empty:
            last_power = float(power_prices.dropna().iloc[-1])
        else:
            last_power = 0.0

        if gas_prices is not None and not gas_prices.empty and heat_rate_gas:
            gas_cost = float(gas_prices.dropna().iloc[-1]) * float(heat_rate_gas)
            carbon_cost_gas = 0.0
            if carbon_prices is not None and not carbon_prices.empty and emissions_factor_gas:
                carbon_cost_gas = float(carbon_prices.dropna().iloc[-1]) * float(emissions_factor_gas)
            spark_spread = last_power - gas_cost - carbon_cost_gas

        if coal_prices is not None and not coal_prices.empty and heat_rate_coal:
            coal_cost = float(coal_prices.dropna().iloc[-1]) * float(heat_rate_coal)
            carbon_cost_coal = 0.0
            if carbon_prices is not None and not carbon_prices.empty and emissions_factor_coal:
                carbon_cost_coal = float(carbon_prices.dropna().iloc[-1]) * float(emissions_factor_coal)
            dark_spread = last_power - coal_cost - carbon_cost_coal

        capacity_mw = fallback_capacity or 0.0
        if capacity_series is not None and not capacity_series.empty:
            capacity_mw = float(capacity_series.dropna().iloc[-1])

        load_mw = 0.0
        if load_series is not None and not load_series.empty:
            load_mw = float(load_series.dropna().iloc[-1])

        utilization = 0.0
        if capacity_mw > 0:
            utilization = max(0.0, min(1.0, load_mw / capacity_mw))

        return {
            "spark_spread_estimate": float(spark_spread),
            "dark_spread_estimate": float(dark_spread),
            "capacity_mw": float(capacity_mw),
            "load_mw": float(load_mw),
            "utilization_ratio": float(utilization),
        }
