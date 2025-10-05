import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

import pandas as pd
from clickhouse_driver import Client as ClickHouseClient
from openlineage.client import OpenLineageClient
from openlineage.client.facet import SqlJobFacet

from platform.shared.data_quality_framework import DataQualityFramework

logger = logging.getLogger(__name__)


def _get_ch() -> ClickHouseClient:
    return ClickHouseClient(
        host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
        port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
        database="market_intelligence",
        settings={"async_insert": 1, "wait_for_async_insert": 0},
    )


def _ol_client() -> OpenLineageClient:
    url = os.getenv("OPENLINEAGE_URL", "http://marquez:5000")
    return OpenLineageClient(url=url)


def run_cross_source_validation_job(hours_back: int = 24) -> None:
    ch = _get_ch()
    ol = _ol_client()
    dq = DataQualityFramework()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours_back)

    # Example: compare Argus vs Platts per instrument
    # Placeholder fetch; in production, query connectors' landing tables or upstream stores
    primary_df = pd.DataFrame()
    secondary_df = pd.DataFrame()

    sources = {"primary": primary_df, "secondary": secondary_df}
    result = dq.validate_cross_source_data(sources)

    # Persist sample row if any differences
    rows: List[List[Any]] = []
    for rec in result.get("differences", []):
        rows.append([
            end_dt,
            rec.get("primary_source", "primary"),
            rec.get("secondary_source", "secondary"),
            rec.get("instrument_id", "UNKNOWN"),
            rec.get("metric_name", "value"),
            float(rec.get("rel_diff", 0.0)),
            1 if rec.get("within_tolerance", True) else 0,
            rec.get("reconciled_value"),
            os.getenv("RUN_ID", "00000000-0000-0000-0000-000000000000"),
        ])

    if rows:
        ch.execute(
            """
            INSERT INTO market_intelligence.cross_source_validation
            (ts, primary_source, secondary_source, instrument_id, metric_name, rel_diff, within_tolerance, reconciled_value, run_id)
            VALUES
            """,
            rows,
            types_check=True,
        )

    # Emit OpenLineage (minimal)
    try:
        ol.create_run(
            job_name="dq_cross_source_validation",
            run_id=os.getenv("RUN_ID", "00000000-0000-0000-0000-000000000000"),
            facets={"sql": SqlJobFacet(query="validate_cross_source")},
        )
    except Exception:
        logger.warning("OpenLineage emit failed", exc_info=True)


def run_outliers_job(hours_back: int = 24) -> None:
    ch = _get_ch()
    dq = DataQualityFramework()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours_back)

    # Placeholder: fetch a sample time series per instrument
    series_df = pd.DataFrame()
    analysis = dq.detect_outliers_by_commodity(series_df, "power")

    issue_rows: List[List[Any]] = []
    for out in analysis.get("outliers", []):
        issue_rows.append([
            end_dt,
            out.get("source", "unknown"),
            out.get("instrument_id", "UNKNOWN"),
            "power",
            "validity",
            "warning",
            "outlier_iforest",
            str(out.get("value")),
            None,
            {"score": out.get("score")},
            os.getenv("RUN_ID", "00000000-0000-0000-0000-000000000000"),
        ])

    if issue_rows:
        ch.execute(
            """
            INSERT INTO market_intelligence.data_quality_issues
            (event_time, source, instrument_id, commodity_type, dimension, severity, rule_id, value, expected, metadata, run_id)
            VALUES
            """,
            issue_rows,
            types_check=True,
        )


def run_imputation_job(days_back: int = 7) -> None:
    ch = _get_ch()
    dq = DataQualityFramework()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    # Placeholder: hourly gaps for a subset
    hourly = pd.DataFrame()
    imputed = dq.impute_missing_data(hourly, commodity_type="power", method="linear", max_gap_hours=6)

    # Persist small sample
    rows: List[List[Any]] = []
    for _, r in imputed.iterrows():
        rows.append([
            r.get("hour_start", end_dt.replace(minute=0, second=0, microsecond=0)),
            r.get("market", "power"),
            r.get("instrument_id", "UNKNOWN"),
            r.get("price_type", "trade"),
            float(r.get("imputed_price", 0.0)),
            r.get("method", "linear"),
            int(r.get("gap_length_hours", 0)),
            os.getenv("RUN_ID", "00000000-0000-0000-0000-000000000000"),
        ])

    if rows:
        ch.execute(
            """
            INSERT INTO market_intelligence.market_price_hourly_imputed
            (hour_start, market, instrument_id, price_type, imputed_price, method, gap_length_hours, run_id)
            VALUES
            """,
            rows,
            types_check=True,
        )


