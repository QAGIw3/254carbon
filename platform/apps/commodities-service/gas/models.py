"""
Natural gas-specific schemas.

Extends shared pricing/flow/storage models with gas domain labels as needed
(e.g., hub label for Henry Hub/basis, LNG facility snapshots, etc.).
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from schemas.pricing import FlowMeasurement, PricePoint, StorageSnapshot


class GasPricePoint(PricePoint):
    hub: Optional[str] = None


class GasStorageReport(StorageSnapshot):
    region_label: Optional[str] = None


class PipelineFlowReading(FlowMeasurement):
    pass


class LNGFacilitySnapshot(BaseModel):
    facility: str
    timestamp: datetime
    feedgas_bcf: float
    utilization_pct: Optional[float] = None
    cargoes_in_queue: Optional[int] = None
    destination_basin: Optional[str] = None
    comment: Optional[str] = None


__all__ = [
    "GasPricePoint",
    "GasStorageReport",
    "PipelineFlowReading",
    "LNGFacilitySnapshot",
]
