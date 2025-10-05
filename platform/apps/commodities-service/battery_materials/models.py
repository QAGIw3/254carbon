"""
Battery materials schemas.

Defines core enumerations and models for battery materials pricing and supply
chain entities. Price points extend the shared schema for client consistency.
"""
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field

from schemas.pricing import PricePoint


class Material(str, Enum):
    LITHIUM_CARBONATE = "lithium_carbonate"
    LITHIUM_HYDROXIDE = "lithium_hydroxide"
    SPODUMENE = "spodumene"
    COBALT = "cobalt"
    NICKEL = "nickel"


class BatteryMaterialPrice(PricePoint):
    material: Material
    exchange: Optional[str] = None
    contract_type: Optional[str] = None


class SupplyChainNode(BaseModel):
    node_id: str
    node_type: str
    location: str
    operator: Optional[str] = None
    stage: str
    capacity_tpy: Optional[float] = None
    material: Material
    status: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


__all__ = [
    "Material",
    "BatteryMaterialPrice",
    "SupplyChainNode",
]
