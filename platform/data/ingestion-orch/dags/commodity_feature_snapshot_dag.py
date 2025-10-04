"""Daily commodity feature snapshot DAG.

Computes spark/dark spread and capacity utilization features for power
instruments and persists daily snapshots to ClickHouse for analytics.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../apps/ml-service")))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["data-engineering@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}


def _categorize_feature(feature_name: str) -> str:
    """Map feature name prefixes to high-level categories."""
    mapping = {
        "spark": "spark",
        "clean_spark": "spark",
        "dark": "dark",
        "clean_dark": "dark",
        "capacity": "capacity",
        "reserve": "capacity",
    }

    for prefix, category in mapping.items():
        if feature_name.startswith(prefix):
            return category
    return "other"


def collect_snapshots(**context):
    """Build daily commodity feature snapshots for power instruments."""
    from feature_engineering import FeatureEngineer

    async def _collect() -> List[Dict[str, float]]:
        engineer = FeatureEngineer()
        pool = await engineer._get_pg_pool()
        async with pool.acquire() as conn:
            instruments = await conn.fetch(
                "SELECT instrument_id, market, attrs FROM pg.instrument WHERE market = 'power'"
            )

        snapshots: List[Dict[str, float]] = []
        for instrument in instruments:
            metrics = engineer._build_generation_features(
                instrument_id=instrument["instrument_id"],
                generation_config=engineer._resolve_generation_config(instrument),
            )
            if not metrics:
                continue

            for name, value in metrics.items():
                snapshots.append(
                    {
                        "instrument_id": instrument["instrument_id"],
                        "feature_name": name,
                        "feature_value": float(value),
                        "feature_category": _categorize_feature(name),
                    }
                )

        if engineer.pg_pool is not None:
            await engineer.pg_pool.close()
        try:
            engineer.ch_client.disconnect()
        except Exception:
            pass

        return snapshots

    snapshots = asyncio.run(_collect())
    context["ti"].xcom_push(key="snapshots", value=snapshots)
    return len(snapshots)


def persist_snapshots(**context):
    """Persist collected feature snapshots into ClickHouse."""
    from clickhouse_driver import Client

    ti = context["ti"]
    snapshots = ti.xcom_pull(task_ids="collect_snapshots", key="snapshots") or []
    if not snapshots:
        return 0

    client = Client(host="clickhouse", port=9000)
    execution_date = context.get("execution_date")
    snapshot_date = execution_date.date() if execution_date else datetime.utcnow().date()

    rows = [
        (
            snapshot_date,
            snapshot["instrument_id"],
            snapshot["feature_name"],
            snapshot["feature_category"],
            snapshot["feature_value"],
        )
        for snapshot in snapshots
    ]

    client.execute(
        """
        INSERT INTO ch.daily_commodity_features
            (snapshot_date, instrument_id, feature_name, feature_category, feature_value)
        VALUES
        """,
        rows,
    )

    try:
        client.disconnect()
    except Exception:
        pass

    return len(rows)


with DAG(
    "commodity_feature_snapshot",
    default_args=DEFAULT_ARGS,
    description="Persist daily spark/dark spread and capacity features to ClickHouse",
    schedule_interval="0 2 * * *",  # Daily at 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "features", "commodity"],
) as dag:

    collect = PythonOperator(
        task_id="collect_snapshots",
        python_callable=collect_snapshots,
        provide_context=True,
    )

    persist = PythonOperator(
        task_id="persist_snapshots",
        python_callable=persist_snapshots,
        provide_context=True,
    )

    collect >> persist
