"""Daily gas storage arbitrage run."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
except Exception:  # pragma: no cover
    CollectorRegistry = Gauge = push_to_gateway = None  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../apps/gas_coal_analytics")))

from platform.shared.data_quality_framework import DataQualityFramework
from gas_coal_analytics.jobs import run_storage_arbitrage_job

DEFAULT_ARGS = {
    "owner": "analytics",
    "email_on_failure": True,
    "email": ["gas-ops@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=20),
}

logger = logging.getLogger(__name__)
PROM_GATEWAY = os.getenv("PROM_PUSH_GATEWAY")


def _push_metric(metric: str, value: float) -> None:
    if not PROM_GATEWAY or not Gauge or not CollectorRegistry or not push_to_gateway:  # pragma: no cover
        return
    try:
        registry = CollectorRegistry()
        gauge = Gauge(metric, "Gas storage arbitrage metric", registry=registry)
        gauge.set(value)
        push_to_gateway(PROM_GATEWAY, job="gas_storage_arbitrage", registry=registry)
    except Exception:  # pragma: no cover
        logger.debug("Failed to push Prometheus metric", exc_info=True)


def _hubs_from_conf(conf: dict | None) -> List[str]:
    if conf and conf.get("hubs"):
        return conf["hubs"]
    return ["HENRY", "DAWN"]


def run_arbitrage(**context) -> int:
    logical_date: datetime = context.get("logical_date") or datetime.utcnow()
    hubs = _hubs_from_conf(getattr(context.get("dag_run"), "conf", None))
    results = run_storage_arbitrage_job(logical_date.date(), hubs)
    dq = DataQualityFramework()
    total_outliers = 0
    for result in results:
        schedule_df = pd.DataFrame([entry.dict() for entry in result.schedule])
        if schedule_df.empty:
            continue
        schedule_df = schedule_df.set_index("date")
        schedule_df["value"] = schedule_df["net_cash_flow"]
        qa = dq.detect_outliers_by_commodity(schedule_df[["value"]], commodity_type="gas", method="z_score")
        if isinstance(qa, dict) and qa.get("outlier_count"):
            total_outliers += int(qa["outlier_count"])
            logger.info("Detected %s schedule anomalies for %s", qa["outlier_count"], result.hub)
    _push_metric("gas_storage_runs", len(results))
    _push_metric("gas_storage_outliers", total_outliers)
    return len(results)


with DAG(
    dag_id="gas_storage_arbitrage",
    description="Compute gas storage arbitrage value per hub",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(2),
    catchup=False,
    max_active_runs=1,
    tags=["gas", "analytics"],
) as dag:
    PythonOperator(
        task_id="run_storage_arbitrage",
        python_callable=run_arbitrage,
        provide_context=True,
    )
