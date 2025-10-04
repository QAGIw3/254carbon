"""Daily HDD/CDD metric generation."""
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
from gas_coal_analytics.jobs import run_hdd_cdd_metrics_job

DEFAULT_ARGS = {
    "owner": "analytics",
    "email_on_failure": True,
    "email": ["monitoring@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

logger = logging.getLogger(__name__)
PROM_GATEWAY = os.getenv("PROM_PUSH_GATEWAY")


def _push_metric(metric: str, value: float) -> None:
    if not PROM_GATEWAY or not Gauge or not CollectorRegistry or not push_to_gateway:  # pragma: no cover
        return
    try:
        registry = CollectorRegistry()
        gauge = Gauge(metric, "HDD/CDD pipeline metric", registry=registry)
        gauge.set(value)
        push_to_gateway(PROM_GATEWAY, job="hdd_cdd_metrics", registry=registry)
    except Exception:  # pragma: no cover
        logger.debug("Prometheus push failed", exc_info=True)


def _regions_from_conf(conf: dict | None) -> List[str]:
    if conf and conf.get("regions"):
        return conf["regions"]
    return ["PJM", "ERCOT", "NYISO", "MIDWEST"]


def generate_metrics(**context) -> int:
    logical_date: datetime = context.get("logical_date") or datetime.utcnow()
    regions = _regions_from_conf(getattr(context.get("dag_run"), "conf", None))
    rows = run_hdd_cdd_metrics_job(logical_date.date(), regions)
    dq = DataQualityFramework()
    anomalies = 0
    for row in rows:
        value = row.get("metric_value", 0)
        if value < 0 or value > 150:
            anomalies += 1
            logger.warning("Out-of-range degree day: %s %s=%s", row.get("entity_id"), row.get("metric_name"), value)
    if anomalies:
        dq.quality_scores["hdd_cdd_anomalies"] = anomalies
    _push_metric("hdd_cdd_rows", len(rows))
    _push_metric("hdd_cdd_anomalies", anomalies)
    return len(rows)


with DAG(
    dag_id="hdd_cdd_metrics",
    description="Generate HDD/CDD metrics per region",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(2),
    catchup=False,
    max_active_runs=1,
    tags=["gas", "weather"],
) as dag:
    PythonOperator(
        task_id="compute_hdd_cdd_metrics",
        python_callable=generate_metrics,
        provide_context=True,
    )
