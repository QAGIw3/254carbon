"""Daily volatility regime classification DAG."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

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


def classify_volatility_regimes(**context) -> int:
    dag_run = context.get("dag_run")
    start_dt: Optional[datetime] = None
    if dag_run and getattr(dag_run, "conf", None):
        start_conf = dag_run.conf.get("start_date")
        if start_conf:
            start_dt = datetime.fromisoformat(start_conf)

    async def _run() -> int:
        pool = await asyncpg.create_pool(PG_DSN)
        async with pool.acquire() as conn:
            records = await conn.fetch("SELECT instrument_id FROM pg.instrument")

        data_access = DataAccessLayer()
        persistence = ResearchPersistence(ch_client=data_access.client)
        framework = CommodityResearchFramework(
            data_access=data_access,
            persistence=persistence,
        )

        processed = 0
        for record in records:
            instrument_id = record["instrument_id"]
            lookback_days = 365
            if start_dt:
                lookback_days = max((datetime.utcnow() - start_dt).days, 90)
            try:
                framework.analyze_volatility_regimes(
                    instrument_id=instrument_id,
                    method="auto",
                    n_regimes=3,
                    lookback_days=lookback_days,
                    persist=True,
                )
                processed += 1
            except ValueError as exc:
                logger.info("Skipping %s: %s", instrument_id, exc)
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("Regime detection failed for %s: %s", instrument_id, exc)

        await pool.close()
        persistence.close()
        data_access.close()
        _emit_metric("research_volatility_regime_processed_total", processed)
        return processed

    return asyncio.run(_run())


with DAG(
    "volatility_regime_classification",
    default_args=DEFAULT_ARGS,
    description="Daily volatility regime classification",
    schedule_interval="0 2 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["research", "volatility"],
) as dag:

    classify = PythonOperator(
        task_id="classify_volatility",
        python_callable=classify_volatility_regimes,
        provide_context=True,
    )
def _emit_metric(metric: str, value: float) -> None:
    if not PROM_GATEWAY or Gauge is None or CollectorRegistry is None or push_to_gateway is None:
        return
    try:
        registry = CollectorRegistry()
        gauge = Gauge(metric, "Commodity research metric", registry=registry)
        gauge.set(value)
        push_to_gateway(PROM_GATEWAY, job="volatility_regime_classification", registry=registry)
    except Exception:  # pragma: no cover
        logger.debug("Prometheus push failed", exc_info=True)
