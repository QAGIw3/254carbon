"""
Shared pricing schemas for the commodities service.

These models represent common shapes used in commodity pricing endpoints,
including generic price points, storage snapshots, flows, and simple latest
snapshots. Domain-specific schemas for each commodity extend these where
useful to add extra fields.
"""
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class PricePoint(BaseModel):
    instrument_id: str
    timestamp: datetime
    price: float
    currency: Optional[str] = None
    unit: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None
    price_type: Optional[str] = None


class StorageSnapshot(BaseModel):
    report_date: date
    region: str
    inventory_bcf: float
    net_change_bcf: float
    year_ago_bcf: float
    five_year_avg_bcf: float
    capacity_bcf: Optional[float] = None


class FlowMeasurement(BaseModel):
    instrument_id: str
    pipeline_name: str
    zone: Optional[str] = None
    timestamp: datetime
    flow_mmcfd: float
    capacity_mmcfd: Optional[float] = None
    utilization_pct: Optional[float] = None
    source: Optional[str] = None


class SnapshotMetadata(BaseModel):
    instrument_id: str
    latest_price: Optional[float]


__all__ = [
    "PricePoint",
    "StorageSnapshot",
    "FlowMeasurement",
    "SnapshotMetadata",
]
