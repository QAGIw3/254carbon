"""
Synthetic fallbacks for natural gas endpoints.

These generators provide realistic-but-fake time-series for local development
and as a safety net when ClickHouse is temporarily unavailable. They mirror
the characteristics of the former standalone gas service to preserve dev
parity and UX expectations.
"""
import math
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional

import numpy as np

from .models import GasPricePoint, GasStorageReport, LNGFacilitySnapshot, PipelineFlowReading


def generate_price_series(
    instrument_id: str,
    start: datetime,
    end: datetime,
    hub: Optional[str] = None,
    currency: str = "USD",
    unit: str = "$/MMBtu",
) -> List[GasPricePoint]:
    """Generate synthetic Henry Hub or basis prices for dev parity."""

    points: List[GasPricePoint] = []
    current = start
    base_price = 3.50 if "HENRY" in instrument_id.upper() else 0.25
    volatility = 0.15 if "HENRY" in instrument_id.upper() else 0.05

    while current <= end:
        seasonal = 1 + 0.25 * math.sin(2 * math.pi * current.timetuple().tm_yday / 365)
        price = base_price * seasonal + np.random.normal(0, volatility)
        points.append(
            GasPricePoint(
                instrument_id=instrument_id,
                timestamp=current,
                price=round(float(max(price, 0.5)), 3),
                currency=currency,
                unit=unit,
                location=hub,
                price_type="spot",
                volume=float(max(np.random.normal(50000, 7500), 10000)),
            )
        )
        current += timedelta(hours=6)
        if len(points) >= 200:
            break

    return points


def generate_storage_reports(
    start: date,
    end: date,
    region: str = "Lower 48",
) -> List[GasStorageReport]:
    """Generate weekly storage data consistent with legacy service."""

    regions = {
        "Lower 48": 4000,
        "East": 1500,
        "Midwest": 1000,
        "Mountain": 200,
        "Pacific": 300,
        "South Central": 1000,
    }
    capacity = regions.get(region, 2500)
    reports: List[GasStorageReport] = []

    current = start
    while current <= end:
        if current.weekday() == 3:  # Thursday
            day = current.timetuple().tm_yday
            seasonal = 0.6 + 0.3 * math.cos(2 * math.pi * (day - 90) / 365)
            inventory = capacity * seasonal
            net_change = np.random.normal(10, 20)
            reports.append(
                GasStorageReport(
                    report_date=current,
                    region=region,
                    inventory_bcf=round(float(max(inventory, 0)), 2),
                    net_change_bcf=round(float(net_change), 2),
                    year_ago_bcf=round(float(capacity * 0.55), 2),
                    five_year_avg_bcf=round(float(capacity * 0.58), 2),
                    capacity_bcf=float(capacity),
                    region_label=region,
                )
            )
        current += timedelta(days=1)
        if len(reports) >= 52:
            break

    return reports


def generate_pipeline_flows(
    pipeline_name: str,
    start: datetime,
    end: datetime,
    zone: Optional[str] = None,
) -> List[PipelineFlowReading]:
    """Generate synthetic pipeline flow utilization metrics."""

    readings: List[PipelineFlowReading] = []
    current = start
    capacity = float(np.random.uniform(5.0, 11.0)) * 1000  # MMSCFD

    while current <= end:
        utilization = max(min(np.random.normal(0.65, 0.1), 0.98), 0.35)
        flow = capacity * utilization
        readings.append(
            PipelineFlowReading(
                instrument_id=f"{pipeline_name}_{zone or 'main'}".lower(),
                pipeline_name=pipeline_name,
                zone=zone,
                timestamp=current,
                flow_mmcfd=round(float(flow), 2),
                capacity_mmcfd=round(float(capacity), 2),
                utilization_pct=round(float(utilization * 100), 2),
            )
        )
        current += timedelta(hours=6)
        if len(readings) >= 120:
            break

    return readings


def generate_lng_feedgas(
    start: datetime,
    end: datetime,
    facility: str = "Sabine Pass",
) -> List[LNGFacilitySnapshot]:
    """Generate synthetic LNG feedgas data."""

    snapshots: List[LNGFacilitySnapshot] = []
    current = start
    base_feedgas = 3.5  # Bcf/d

    while current <= end:
        variation = np.random.normal(0, 0.2)
        feedgas = max(base_feedgas + variation, 1.0)
        utilization = min(feedgas / base_feedgas * 100, 105)
        snapshots.append(
            LNGFacilitySnapshot(
                facility=facility,
                timestamp=current,
                feedgas_bcf=round(float(feedgas), 3),
                utilization_pct=round(float(utilization), 2),
                cargoes_in_queue=max(int(np.random.poisson(1)), 0),
            )
        )
        current += timedelta(hours=12)
        if len(snapshots) >= 60:
            break

    return snapshots


__all__ = [
    "generate_price_series",
    "generate_storage_reports",
    "generate_pipeline_flows",
    "generate_lng_feedgas",
]
