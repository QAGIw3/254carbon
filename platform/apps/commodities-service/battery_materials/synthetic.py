"""
Synthetic fallbacks for battery material endpoints.

Generates simple daily lithium series and a sample supply chain snapshot
so routes work in isolation during development or when CH is unavailable.
"""
from datetime import date, datetime, timedelta
from typing import List

import numpy as np

from .models import BatteryMaterialPrice, Material, SupplyChainNode


def generate_lithium_prices(
    material: Material,
    start: date,
    end: date,
    base_price: float,
    unit: str,
    exchange: str,
) -> List[BatteryMaterialPrice]:
    points: List[BatteryMaterialPrice] = []
    current = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    while current <= end_dt:
        drift = np.random.normal(0, base_price * 0.01)
        price = max(base_price + drift, base_price * 0.4)
        points.append(
            BatteryMaterialPrice(
                instrument_id=f"{material.value}_{exchange}".upper(),
                material=material,
                timestamp=current,
                price=round(float(price), 2),
                currency="USD",
                unit=unit,
                exchange=exchange,
                contract_type="spot",
            )
        )
        current += timedelta(days=1)
        if len(points) >= 120:
            break

    return points


def sample_supply_chain() -> List[SupplyChainNode]:
    return [
        SupplyChainNode(
            node_id="MINE-GREENBUSHES",
            node_type="mine",
            location="Western Australia",
            operator="Tianqi / Albemarle",
            stage="extraction",
            capacity_tpy=1500000,
            material=Material.SPODUMENE,
            status="operating",
            metadata={"grade_pct": "2.8"},
        ),
        SupplyChainNode(
            node_id="REF-QUINAN",
            node_type="refinery",
            location="Sichuan, China",
            operator="Ganfeng",
            stage="refining",
            capacity_tpy=80000,
            material=Material.LITHIUM_CARBONATE,
            status="operating",
            metadata={"emissions_intensity": "7.2 tCO2e/t"},
        ),
        SupplyChainNode(
            node_id="CATHODE-KOREA",
            node_type="cathode",
            location="South Korea",
            operator="LG Chem",
            stage="manufacturing",
            capacity_tpy=120000,
            material=Material.LITHIUM_HYDROXIDE,
            status="operating",
            metadata={"chemistry": "NMC811"},
        ),
    ]


__all__ = ["generate_lithium_prices", "sample_supply_chain"]
