"""Pydantic models for gas and coal analytics service."""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class StorageArbitrageScheduleEntry(BaseModel):
    """Single-day storage action recommendation."""
    date: date
    action: str
    volume_mmbtu: float
    inventory_mmbtu: float
    price: float
    net_cash_flow: float


class StorageArbitrageResult(BaseModel):
    """Storage optimization analytics output."""
    as_of_date: date
    hub: str
    region: Optional[str]
    expected_storage_value: float
    breakeven_spread: Optional[float]
    schedule: List[StorageArbitrageScheduleEntry]
    cost_parameters: Dict[str, float]
    constraint_summary: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class WeatherImpactCoefficient(BaseModel):
    """Weather beta for HDD/CDD impacts."""
    as_of_date: date
    entity_id: str
    coef_type: str
    coefficient: float
    r2: Optional[float]
    window: str
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class CoalToGasSwitchingResult(BaseModel):
    """Coal-to-gas switching economics summary."""
    as_of_date: date
    region: str
    coal_cost_mwh: float
    gas_cost_mwh: float
    co2_price: float
    breakeven_gas_price: float
    switch_share: float
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class GasBasisModelResult(BaseModel):
    """Regional basis model prediction."""
    as_of_date: date
    hub: str
    predicted_basis: float
    actual_basis: Optional[float]
    method: str
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    feature_snapshot: Dict[str, Any] = Field(default_factory=dict)
