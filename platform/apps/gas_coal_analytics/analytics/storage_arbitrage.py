"""Storage arbitrage optimization for natural gas assets."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..clients.clickhouse import query_dataframe
from ..models import StorageArbitrageResult, StorageArbitrageScheduleEntry

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    hub: str
    region: str
    capacity_mmbtu: float
    min_inventory_pct: float
    initial_inventory_pct: float
    max_injection_rate_mmbtu: float
    max_withdrawal_rate_mmbtu: float
    inject_cost_per_mmbtu: float
    withdraw_cost_per_mmbtu: float
    carry_cost_per_mmbtu_day: float
    shrinkage_pct: float = 0.0


DEFAULT_STORAGE_CONFIGS: Dict[str, StorageConfig] = {
    "HENRY": StorageConfig(
        hub="HENRY",
        region="US_GULF",
        capacity_mmbtu=2_500_000.0,
        min_inventory_pct=0.1,
        initial_inventory_pct=0.55,
        max_injection_rate_mmbtu=60_000.0,
        max_withdrawal_rate_mmbtu=70_000.0,
        inject_cost_per_mmbtu=0.05,
        withdraw_cost_per_mmbtu=0.04,
        carry_cost_per_mmbtu_day=0.002,
        shrinkage_pct=0.004,
    ),
    "DAWN": StorageConfig(
        hub="DAWN",
        region="EAST_CANADA",
        capacity_mmbtu=1_600_000.0,
        min_inventory_pct=0.15,
        initial_inventory_pct=0.45,
        max_injection_rate_mmbtu=45_000.0,
        max_withdrawal_rate_mmbtu=50_000.0,
        inject_cost_per_mmbtu=0.06,
        withdraw_cost_per_mmbtu=0.05,
        carry_cost_per_mmbtu_day=0.0025,
        shrinkage_pct=0.006,
    ),
}


class StorageArbitrageCalculator:
    """Calculate storage arbitrage strategy using greedy thresholds."""

    def __init__(self, storage_config: Optional[StorageConfig] = None):
        self.storage_config = storage_config

    def _load_forward_curve(self, hub: str, as_of: date) -> pd.DataFrame:
        sql = """
            SELECT
                delivery_start AS date,
                price
            FROM ch.forward_curve_points
            WHERE instrument_id = %(hub)s
              AND as_of_date = %(as_of)s
              AND tenor_type IN ('DAILY', 'GAS_DAY')
            ORDER BY delivery_start
            LIMIT 365
        """
        df = query_dataframe(sql, {"hub": hub, "as_of": as_of})
        if df.empty:
            logger.warning("No forward curve found for %s on %s; generating synthetic curve", hub, as_of)
            days = pd.date_range(as_of, periods=180, freq="D")
            base_price = 3.25
            seasonal_component = 0.5 * np.sin(np.linspace(0, 4 * np.pi, len(days)))
            trend_component = np.linspace(0, 0.5, len(days))
            synthetic_price = base_price + seasonal_component + trend_component
            df = pd.DataFrame({"date": days.date, "price": synthetic_price})
        return df

    def _load_inventory_pct(self, hub: str, as_of: date) -> Optional[float]:
        sql = """
            SELECT
                metric_value
            FROM ch.supply_demand_metrics
            WHERE entity_id = %(hub)s
              AND metric_name = 'ng_storage_pct_full'
              AND date <= %(as_of)s
            ORDER BY date DESC
            LIMIT 1
        """
        df = query_dataframe(sql, {"hub": hub, "as_of": as_of})
        if df.empty:
            return None
        return float(df.iloc[0]["metric_value"])

    def _resolve_config(self, hub: str, as_of: date) -> StorageConfig:
        if self.storage_config:
            return self.storage_config
        config = DEFAULT_STORAGE_CONFIGS.get(hub.upper())
        if not config:
            config = replace(DEFAULT_STORAGE_CONFIGS["HENRY"], hub=hub.upper())
        inventory_pct = self._load_inventory_pct(hub, as_of)
        if inventory_pct is not None:
            config = replace(config, initial_inventory_pct=inventory_pct)
        return config

    def _build_schedule(self, curve: pd.DataFrame, cfg: StorageConfig) -> List[StorageArbitrageScheduleEntry]:
        price_series = curve["price"].astype(float)
        low_price_threshold = price_series.quantile(0.25)
        high_price_threshold = price_series.quantile(0.75)
        inventory = cfg.capacity_mmbtu * cfg.initial_inventory_pct
        min_inventory = cfg.capacity_mmbtu * cfg.min_inventory_pct
        schedule: List[StorageArbitrageScheduleEntry] = []
        total_cash = 0.0
        total_volume_traded = 0.0

        for _, row in curve.iterrows():
            day: date = row["date"]
            price: float = float(row["price"])
            action = "HOLD"
            volume = 0.0
            net_cash_flow = -inventory * cfg.carry_cost_per_mmbtu_day

            if price <= low_price_threshold and inventory < cfg.capacity_mmbtu:
                available_space = cfg.capacity_mmbtu - inventory
                volume = min(cfg.max_injection_rate_mmbtu, available_space)
                action = "INJECT"
                inventory += volume * (1 - cfg.shrinkage_pct)
                net_cash_flow -= volume * (price + cfg.inject_cost_per_mmbtu)
            elif price >= high_price_threshold and inventory > min_inventory:
                available_inventory = inventory - min_inventory
                volume = min(cfg.max_withdrawal_rate_mmbtu, available_inventory)
                action = "WITHDRAW"
                inventory -= volume
                net_cash_flow += volume * (price - cfg.withdraw_cost_per_mmbtu)

            total_cash += net_cash_flow
            if action != "HOLD":
                total_volume_traded += volume
            schedule.append(
                StorageArbitrageScheduleEntry(
                    date=day,
                    action=action,
                    volume_mmbtu=volume,
                    inventory_mmbtu=inventory,
                    price=price,
                    net_cash_flow=net_cash_flow,
                )
            )

        breakeven_spread = float(high_price_threshold - low_price_threshold - (
            cfg.inject_cost_per_mmbtu + cfg.withdraw_cost_per_mmbtu
        ))

        diagnostics = {
            "mean_price": float(price_series.mean()),
            "std_price": float(price_series.std()),
            "low_threshold": float(low_price_threshold),
            "high_threshold": float(high_price_threshold),
            "total_volume_mmbtu": total_volume_traded,
        }

        constraint_summary = {
            "max_inject": cfg.max_injection_rate_mmbtu,
            "max_withdraw": cfg.max_withdrawal_rate_mmbtu,
            "min_inventory": min_inventory,
            "capacity": cfg.capacity_mmbtu,
        }

        return schedule, total_cash, breakeven_spread, diagnostics, constraint_summary

    def compute(self, hub: str, as_of: date) -> StorageArbitrageResult:
        curve = self._load_forward_curve(hub, as_of)
        cfg = self._resolve_config(hub, as_of)
        schedule, total_cash, breakeven_spread, diagnostics, constraint_summary = self._build_schedule(curve, cfg)
        cost_parameters = {
            "inject_cost_per_mmbtu": cfg.inject_cost_per_mmbtu,
            "withdraw_cost_per_mmbtu": cfg.withdraw_cost_per_mmbtu,
            "carry_cost_per_mmbtu_day": cfg.carry_cost_per_mmbtu_day,
            "shrinkage_pct": cfg.shrinkage_pct,
        }
        diagnostics = {**diagnostics, "expected_value_musd": total_cash / 1_000_000}
        return StorageArbitrageResult(
            as_of_date=as_of,
            hub=cfg.hub,
            region=cfg.region,
            expected_storage_value=total_cash,
            breakeven_spread=breakeven_spread,
            schedule=schedule,
            cost_parameters=cost_parameters,
            constraint_summary=constraint_summary,
            diagnostics=diagnostics,
        )

    def persist(self, result: StorageArbitrageResult) -> None:
        from ..clients.clickhouse import insert_rows

        payload = [{
            "as_of_date": result.as_of_date,
            "hub": result.hub,
            "region": result.region,
            "curve_reference": f"{result.hub}_CURVE",
            "expected_storage_value": result.expected_storage_value,
            "breakeven_spread": result.breakeven_spread,
            "optimal_schedule": json.dumps([entry.dict() for entry in result.schedule]),
            "cost_parameters": json.dumps(result.cost_parameters),
            "constraint_summary": json.dumps(result.constraint_summary),
            "diagnostics": json.dumps(result.diagnostics),
            "model_version": "v1",
        }]
        insert_rows("market_intelligence.gas_storage_arbitrage", payload)
