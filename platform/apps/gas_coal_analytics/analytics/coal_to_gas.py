"""Coal-to-gas switching economics calculator."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from ..clients.clickhouse import query_dataframe, insert_rows
from ..models import CoalToGasSwitchingResult

logger = logging.getLogger(__name__)


@dataclass
class PlantAttributes:
    name: str
    heat_rate_mmbtu_mwh: float
    vom_usd_mwh: float
    emissions_ton_mwh: float
    capacity_mw: float


@dataclass
class RegionConfig:
    region: str
    gas_hub: str
    coal_index: str
    coal_delivery_adder: float
    gas_basis_adjustment: float
    gas_heat_rate: float
    gas_vom: float
    gas_emissions: float
    plants: List[PlantAttributes]


REGION_CONFIGS: Dict[str, RegionConfig] = {
    "PJM": RegionConfig(
        region="PJM",
        gas_hub="DOMINION_SOUTH",
        coal_index="API2",
        coal_delivery_adder=1.15,
        gas_basis_adjustment=0.3,
        gas_heat_rate=7.3,
        gas_vom=3.5,
        gas_emissions=0.37,
        plants=[
            PlantAttributes("Coal_Subcritical", 10.2, 4.0, 0.98, 12000),
            PlantAttributes("Coal_Supercritical", 9.6, 3.8, 0.92, 8000),
            PlantAttributes("Coal_Modern", 9.1, 3.5, 0.88, 6000),
        ],
    ),
    "ERCOT": RegionConfig(
        region="ERCOT",
        gas_hub="HOUSTON_SHIP_CHANNEL",
        coal_index="PRB",
        coal_delivery_adder=1.25,
        gas_basis_adjustment=0.2,
        gas_heat_rate=7.1,
        gas_vom=3.0,
        gas_emissions=0.36,
        plants=[
            PlantAttributes("Coal_Lignite", 10.8, 4.5, 1.02, 10000),
            PlantAttributes("Coal_Subcritical", 10.1, 4.2, 0.97, 7000),
        ],
    ),
}


class CoalToGasSwitchingCalculator:
    """Estimate breakeven gas price and switching share by region."""

    def __init__(self, co2_price_default: float = 15.0):
        self.co2_price_default = co2_price_default

    def _load_price(self, instrument_id: str, as_of: date) -> float:
        sql = """
            SELECT
                avg_price
            FROM ch.market_price_daily_agg
            WHERE instrument_id = %(instrument)s
              AND date <= %(as_of)s
            ORDER BY date DESC
            LIMIT 1
        """
        df = query_dataframe(sql, {"instrument": instrument_id, "as_of": as_of})
        if df.empty:
            logger.warning("Missing price for %s on %s; using synthetic fallback", instrument_id, as_of)
            return 2.75 if "gas" in instrument_id.lower() else 45.0
        return float(df.iloc[0]["avg_price"])

    def _resolve_config(self, region: str) -> RegionConfig:
        cfg = REGION_CONFIGS.get(region.upper())
        if not cfg:
            logger.warning("Region %s not configured; defaulting to PJM parameters", region)
            base = REGION_CONFIGS["PJM"]
            cfg = replace(base, region=region.upper(), plants=list(base.plants))
        return cfg

    def compute(self, region: str, as_of: date, co2_price: Optional[float] = None) -> CoalToGasSwitchingResult:
        cfg = self._resolve_config(region)
        gas_price = self._load_price(cfg.gas_hub, as_of) + cfg.gas_basis_adjustment
        coal_price = self._load_price(cfg.coal_index, as_of) * cfg.coal_delivery_adder
        co2 = co2_price if co2_price is not None else self.co2_price_default

        coal_costs = []
        breakevens = []
        switch_capacity = 0.0
        total_capacity = 0.0

        gas_cost_mwh = gas_price * cfg.gas_heat_rate + cfg.gas_vom + co2 * cfg.gas_emissions

        for plant in cfg.plants:
            coal_cost = coal_price * plant.heat_rate_mmbtu_mwh + plant.vom_usd_mwh + co2 * plant.emissions_ton_mwh
            breakeven = (coal_cost - cfg.gas_vom - co2 * (cfg.gas_emissions - plant.emissions_ton_mwh)) / max(cfg.gas_heat_rate, 0.01)
            coal_costs.append(coal_cost)
            breakevens.append(breakeven)
            total_capacity += plant.capacity_mw
            if gas_price <= breakeven:
                switch_capacity += plant.capacity_mw

        coal_cost_avg = float(np.average(coal_costs, weights=[p.capacity_mw for p in cfg.plants])) if coal_costs else 0.0
        breakeven_avg = float(np.average(breakevens, weights=[p.capacity_mw for p in cfg.plants])) if breakevens else 0.0
        switch_share = float(switch_capacity / total_capacity) if total_capacity else 0.0
        diagnostics = {
            "plant_count": len(cfg.plants),
            "gas_price": gas_price,
            "coal_price": coal_price,
            "capacity_weighted_coal_cost": coal_cost_avg,
            "capacity_switchable_mw": switch_capacity,
        }

        result = CoalToGasSwitchingResult(
            as_of_date=as_of,
            region=cfg.region,
            coal_cost_mwh=coal_cost_avg,
            gas_cost_mwh=gas_cost_mwh,
            co2_price=co2,
            breakeven_gas_price=breakeven_avg,
            switch_share=switch_share,
            diagnostics=diagnostics,
        )
        return result

    def persist(self, result: CoalToGasSwitchingResult) -> None:
        row = {
            "as_of_date": result.as_of_date,
            "region": result.region,
            "coal_cost_mwh": result.coal_cost_mwh,
            "gas_cost_mwh": result.gas_cost_mwh,
            "co2_price": result.co2_price,
            "breakeven_gas_price": result.breakeven_gas_price,
            "switch_share": result.switch_share,
            "diagnostics": json.dumps(result.diagnostics),
            "model_version": "v1",
        }
        insert_rows("market_intelligence.coal_gas_switching", [row])
