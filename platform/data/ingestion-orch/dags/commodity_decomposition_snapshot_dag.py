"""Daily commodity decomposition snapshot DAG."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import asyncpg
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

try:  # Optional Prometheus metrics
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
except Exception:  # pragma: no cover - optional dependency
    CollectorRegistry = None  # type: ignore
    Gauge = None  # type: ignore
    push_to_gateway = None  # type: ignore

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../apps/ml-service")))

from commodity_research_framework import CommodityResearchFramework
from data_access import DataAccessLayer
from research_persistence import ResearchPersistence


DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["data-engineering@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

PG_DSN = os.getenv(
    "MARKET_INTELLIGENCE_PG_DSN",
    "postgresql://postgres:postgres@postgres:5432/market_intelligence",
)
PROM_GATEWAY = os.getenv("PROM_PUSH_GATEWAY")

logger = logging.getLogger(__name__)


def _infer_commodity_type(record: Dict[str, any]) -> str:
    attrs = record.get("attrs") or {}
    return attrs.get("commodity_type") or (record.get("market") or "other")


def run_decomposition_snapshot(**context) -> int:
    dag_run = context.get("dag_run")
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    if dag_run and getattr(dag_run, "conf", None):
        start_conf = dag_run.conf.get("start_date")
        end_conf = dag_run.conf.get("end_date")
        if start_conf:
            start_dt = datetime.fromisoformat(start_conf)
        if end_conf:
            end_dt = datetime.fromisoformat(end_conf)

    async def _run() -> int:
        pool = await asyncpg.create_pool(PG_DSN)
        async with pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT instrument_id, market, attrs FROM pg.instrument"
            )

        data_access = DataAccessLayer()
        persistence = ResearchPersistence(ch_client=data_access.client)
        framework = CommodityResearchFramework(
            data_access=data_access,
            persistence=persistence,
        )

        processed = 0
        for record in records:
            instrument_id = record["instrument_id"]
            commodity_type = _infer_commodity_type(dict(record))
            try:
                framework.generate_time_series_decomposition(
                    instrument_id=instrument_id,
                    commodity_type=commodity_type,
                    start=start_dt,
                    end=end_dt,
                    persist=True,
                )
                processed += 1
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning(
                    "Decomposition failed for %s (%s): %s",
                    instrument_id,
                    commodity_type,
                    exc,
                )

        await pool.close()
        persistence.close()
        data_access.close()
        _emit_metric("research_decomposition_processed_total", processed)
        return processed

    return asyncio.run(_run())


with DAG(
    "commodity_decomposition_snapshot",
    default_args=DEFAULT_ARGS,
    description="Daily decomposition snapshots for commodity instruments",
    schedule_interval="30 1 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["research", "decomposition"],
) as dag:

    snapshot = PythonOperator(
        task_id="run_decomposition_snapshot",
        python_callable=run_decomposition_snapshot,
        provide_context=True,
    )

def _emit_metric(metric: str, value: float) -> None:
    if not PROM_GATEWAY or Gauge is None or CollectorRegistry is None or push_to_gateway is None:
        return
    try:
        registry = CollectorRegistry()
        gauge = Gauge(metric, "Commodity research metric", registry=registry)
        gauge.set(value)
        push_to_gateway(PROM_GATEWAY, job="commodity_decomposition_snapshot", registry=registry)
    except Exception:  # pragma: no cover - best effort
        logger.debug("Prometheus push failed", exc_info=True)
