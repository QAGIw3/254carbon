"""
Gas commodity endpoints.

Endpoints
- GET /api/v1/commodities/gas/prices: Henry Hub/basis price series
- GET /api/v1/commodities/gas/storage: EIA storage-like weekly reports
- GET /api/v1/commodities/gas/pipelines: Pipeline flows and utilization
- GET /api/v1/commodities/gas/lng: LNG feedgas snapshots

Implementation notes
- Prefers ClickHouse reads with simple query fragments; on error or no rows,
  falls back to deterministic synthetic series for dev parity.
- Uses Redis caching with TTL profiles aligned to data dynamics.
- Requires role `gas_data_access` via dependency for defense-in-depth.
"""
import logging
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from deps.auth import require_roles
from deps.cache import CacheTTL, cache_response
from deps.db import fetch_clickhouse
from deps.query_builders import build_price_query
from .models import (
    GasPricePoint,
    GasStorageReport,
    LNGFacilitySnapshot,
    PipelineFlowReading,
)
from .synthetic import (
    generate_lng_feedgas,
    generate_pipeline_flows,
    generate_price_series,
    generate_storage_reports,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/commodities/gas", tags=["gas"])


def _gas_price_cache_key(
    instrument_id: str,
    start_date: date,
    end_date: date,
    price_type: Optional[str] = None,
    location: Optional[str] = None,
    **_: object,
) -> str:
    # Compose a stable cache key from relevant query parameters. The decorator
    # will namespace it and hash the full set of args.
    return ";".join(
        [
            instrument_id,
            start_date.isoformat(),
            end_date.isoformat(),
            price_type or "",
            location or "",
        ]
    )


@router.get("/prices", response_model=List[GasPricePoint])
@cache_response("gas.prices", CacheTTL.REALTIME, key_builder=_gas_price_cache_key)
async def get_gas_prices(
    instrument_id: str = Query(
        "NG_HENRY_HUB",
        description="Instrument identifier e.g. NG_HENRY_HUB",
    ),
    start_date: date = Query(..., description="Start date (inclusive)"),
    end_date: date = Query(..., description="End date (inclusive)"),
    price_type: Optional[str] = Query(None, description="Optional price type filter"),
    location: Optional[str] = Query(None, description="Optional location filter"),
    limit: int = Query(500, le=5000, description="Maximum rows to return"),
    user: dict = Depends(require_roles("gas_data_access")),
) -> List[GasPricePoint]:
    """Return Henry Hub and regional gas prices."""

    del user  # Authorization handled via dependency

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    try:
        query, params = build_price_query(
            instrument_id=instrument_id,
            start=start_dt,
            end=end_dt,
            price_type=price_type,
            location=location,
            limit=limit,
        )
        rows = await fetch_clickhouse(query, params)
    except Exception as exc:  # pragma: no cover - ClickHouse unavailable
        logger.warning("ClickHouse unavailable for gas prices: %s", exc)
        rows = []

    if rows:
        return [
            GasPricePoint(
                instrument_id=row["instrument_id"],
                timestamp=row["event_time"],
                price=float(row["value"]),
                currency=row.get("currency"),
                unit=row.get("unit"),
                location=row.get("location_code"),
                source=row.get("source"),
                price_type=row.get("price_type"),
                hub=row.get("location_code"),
                volume=float(row["volume"]) if row.get("volume") is not None else None,
            )
            for row in rows
        ]

    logger.info("No ClickHouse data for %s – returning synthetic fallback", instrument_id)
    # No data available (or ClickHouse down): provide synthetic fallback.
    return generate_price_series(
        instrument_id=instrument_id,
        start=start_dt,
        end=end_dt,
        hub=location or instrument_id,
    )


def _storage_cache_key(
    start_date: date,
    end_date: date,
    region: str = "Lower 48",
    **_: object,
) -> str:
    return f"{region}:{start_date.isoformat()}:{end_date.isoformat()}"


@router.get("/storage", response_model=List[GasStorageReport])
@cache_response("gas.storage", CacheTTL.SEMI_STATIC, key_builder=_storage_cache_key)
async def get_storage_reports(
    start_date: date = Query(..., description="Start date for storage reports"),
    end_date: date = Query(..., description="End date for storage reports"),
    region: str = Query("Lower 48", description="EIA storage region"),
    user: dict = Depends(require_roles("gas_data_access")),
) -> List[GasStorageReport]:
    """Return weekly EIA storage reports."""

    del user

    try:
        query = """
        SELECT
            event_time AS report_date,
            location_code AS region,
            value AS inventory_bcf,
            metadata:net_change AS net_change_bcf,
            metadata:year_ago AS year_ago_bcf,
            metadata:five_year_avg AS five_year_avg_bcf,
            metadata:capacity AS capacity_bcf
        FROM market_intelligence.market_price_ticks
        WHERE product = 'storage'
          AND location_code = %(region)s
          AND event_time >= %(start)s
          AND event_time <= %(end)s
        ORDER BY event_time
        """
        rows = await fetch_clickhouse(
            query,
            {
                "region": region,
                "start": datetime.combine(start_date, datetime.min.time()),
                "end": datetime.combine(end_date, datetime.max.time()),
            },
        )
    except Exception as exc:  # pragma: no cover - ClickHouse unavailable
        logger.warning("ClickHouse unavailable for storage: %s", exc)
        rows = []

    if rows:
        return [
            GasStorageReport(
                report_date=row["report_date"],
                region=row.get("region", region),
                inventory_bcf=float(row["inventory_bcf"]),
                net_change_bcf=float(row.get("net_change_bcf") or 0.0),
                year_ago_bcf=float(row.get("year_ago_bcf") or 0.0),
                five_year_avg_bcf=float(row.get("five_year_avg_bcf") or 0.0),
                capacity_bcf=float(row.get("capacity_bcf") or 0.0),
                region_label=row.get("region", region),
            )
            for row in rows
        ]

    return generate_storage_reports(start=start_date, end=end_date, region=region)


def _pipeline_cache_key(
    pipeline: str,
    start_date: date,
    end_date: date,
    zone: Optional[str] = None,
    **_: object,
) -> str:
    return f"{pipeline}:{zone or 'all'}:{start_date.isoformat()}:{end_date.isoformat()}"


@router.get("/pipelines", response_model=List[PipelineFlowReading])
@cache_response("gas.pipelines", CacheTTL.REALTIME, key_builder=_pipeline_cache_key)
async def get_pipeline_flows(
    pipeline: str = Query("transco", description="Pipeline identifier"),
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    zone: Optional[str] = Query(None, description="Optional zone identifier"),
    user: dict = Depends(require_roles("gas_data_access")),
) -> List[PipelineFlowReading]:
    """Return pipeline flow and utilization metrics."""

    del user

    try:
        query = """
        SELECT
            event_time AS timestamp,
            instrument_id,
            location_code AS zone,
            metadata:pipeline_name AS pipeline_name,
            value AS flow_mmcfd,
            metadata:capacity AS capacity_mmcfd,
            metadata:utilization AS utilization_pct,
            source
        FROM market_intelligence.market_price_ticks
        WHERE product = 'pipeline_flow'
          AND metadata:pipeline_name = %(pipeline)s
          AND event_time >= %(start)s
          AND event_time <= %(end)s
        ORDER BY event_time
        """
        params = {
            "pipeline": pipeline,
            "start": datetime.combine(start_date, datetime.min.time()),
            "end": datetime.combine(end_date, datetime.max.time()),
        }
        rows = await fetch_clickhouse(query, params)
    except Exception as exc:  # pragma: no cover
        logger.warning("ClickHouse unavailable for pipeline flow: %s", exc)
        rows = []

    if rows:
        filtered = [row for row in rows if zone is None or row.get("zone") == zone]
        if filtered:
            return [
                PipelineFlowReading(
                    instrument_id=row["instrument_id"],
                    pipeline_name=row.get("pipeline_name", pipeline),
                    zone=row.get("zone"),
                    timestamp=row["timestamp"],
                    flow_mmcfd=float(row.get("flow_mmcfd") or 0.0),
                    capacity_mmcfd=float(row.get("capacity_mmcfd") or 0.0),
                    utilization_pct=float(row.get("utilization_pct") or 0.0),
                    source=row.get("source"),
                )
                for row in filtered
            ]

    return generate_pipeline_flows(
        pipeline_name=pipeline,
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date, datetime.max.time()),
        zone=zone,
    )


def _lng_cache_key(
    facility: str,
    start_date: date,
    end_date: date,
    **_: object,
) -> str:
    return f"{facility}:{start_date.isoformat()}:{end_date.isoformat()}"


@router.get("/lng", response_model=List[LNGFacilitySnapshot])
@cache_response("gas.lng", CacheTTL.SEMI_STATIC, key_builder=_lng_cache_key)
async def get_lng_feedgas(
    facility: str = Query("Sabine Pass", description="LNG facility name"),
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    user: dict = Depends(require_roles("gas_data_access")),
) -> List[LNGFacilitySnapshot]:
    """Return LNG feedgas utilisation metrics."""

    del user

    try:
        query = """
        SELECT
            event_time AS timestamp,
            metadata:facility AS facility,
            value AS feedgas_bcf,
            metadata:utilization AS utilization_pct,
            metadata:cargoes AS cargoes_in_queue,
            metadata:destination AS destination_basin
        FROM market_intelligence.market_price_ticks
        WHERE product = 'lng_feedgas'
          AND metadata:facility = %(facility)s
          AND event_time >= %(start)s
          AND event_time <= %(end)s
        ORDER BY event_time
        """
        params = {
            "facility": facility,
            "start": datetime.combine(start_date, datetime.min.time()),
            "end": datetime.combine(end_date, datetime.max.time()),
        }
        rows = await fetch_clickhouse(query, params)
    except Exception as exc:  # pragma: no cover
        logger.warning("ClickHouse unavailable for LNG feedgas: %s", exc)
        rows = []

    if rows:
        return [
            LNGFacilitySnapshot(
                facility=row.get("facility", facility),
                timestamp=row["timestamp"],
                feedgas_bcf=float(row.get("feedgas_bcf") or 0.0),
                utilization_pct=float(row.get("utilization_pct") or 0.0),
                cargoes_in_queue=int(row.get("cargoes_in_queue") or 0),
                destination_basin=row.get("destination_basin"),
            )
            for row in rows
        ]

    return generate_lng_feedgas(
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date, datetime.max.time()),
        facility=facility,
    )


__all__ = ["router"]
