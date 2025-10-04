"""Gas basis model fitting DAG."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List

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
from gas_coal_analytics.jobs import run_basis_model_job

DEFAULT_ARGS = {
    "owner": "analytics",
    "email_on_failure": True,
    "email": ["analytics-alerts@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

logger = logging.getLogger(__name__)
PROM_GATEWAY = os.getenv("PROM_PUSH_GATEWAY")


def _push_metric(metric: str, value: float) -> None:
    if not PROM_GATEWAY or not Gauge or not CollectorRegistry or not push_to_gateway:  # pragma: no cover
        return
    try:
        registry = CollectorRegistry()
        gauge = Gauge(metric, "Gas basis model metric", registry=registry)
        gauge.set(value)
        push_to_gateway(PROM_GATEWAY, job="gas_basis_model", registry=registry)
    except Exception:  # pragma: no cover
        logger.debug("Prometheus push failed", exc_info=True)


def _hubs(conf: dict | None) -> List[str]:
    if conf and conf.get("hubs"):
        return conf["hubs"]
    return ["HENRY", "DAWN", "CHICAGO"]


def run_basis_models(**context) -> int:
    logical_date: datetime = context.get("logical_date") or datetime.utcnow()
    hubs = _hubs(getattr(context.get("dag_run"), "conf", None))
    results = run_basis_model_job(logical_date.date(), hubs)
    dq = DataQualityFramework()
    low_r2 = 0
    for result in results:
        r2 = float(result.diagnostics.get("r2", 0.0))
        if r2 < 0.2:
            low_r2 += 1
            logger.warning("Low R2 %.3f for hub %s", r2, result.hub)
    if results:
        dq.quality_scores["gas_basis_avg_r2"] = sum(
            float(res.diagnostics.get("r2", 0.0)) for res in results
        ) / len(results)
    _push_metric("gas_basis_models", len(results))
    _push_metric("gas_basis_low_r2", low_r2)
    return len(results)


with DAG(
    dag_id="basis_model_fit",
    description="Fit regional gas basis models",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(2),
    catchup=False,
    max_active_runs=1,
    tags=["gas", "analytics"],
) as dag:
    PythonOperator(
        task_id="fit_gas_basis_models",
        python_callable=run_basis_models,
        provide_context=True,
    )
