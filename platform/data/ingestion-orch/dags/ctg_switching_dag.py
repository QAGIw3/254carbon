"""Coal-to-gas switching analytics DAG."""
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
from gas_coal_analytics.jobs import run_coal_to_gas_switch_job

DEFAULT_ARGS = {
    "owner": "analytics",
    "email_on_failure": True,
    "email": ["power-markets@254carbon.ai"],
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
        gauge = Gauge(metric, "Coal-to-gas switching metric", registry=registry)
        gauge.set(value)
        push_to_gateway(PROM_GATEWAY, job="coal_to_gas_switch", registry=registry)
    except Exception:  # pragma: no cover
        logger.debug("Prometheus push failed", exc_info=True)


def _regions(conf: dict | None) -> List[str]:
    if conf and conf.get("regions"):
        return conf["regions"]
    return ["PJM", "ERCOT"]


def run_switching(**context) -> int:
    logical_date: datetime = context.get("logical_date") or datetime.utcnow()
    regions = _regions(getattr(context.get("dag_run"), "conf", None))
    results = run_coal_to_gas_switch_job(logical_date.date(), regions)
    dq = DataQualityFramework()
    invalid = 0
    for res in results:
        share = res.switch_share
        if share < 0 or share > 1:
            invalid += 1
            logger.warning("Invalid switch share %.3f for %s", share, res.region)
    dq.quality_scores["ctg_valid_regions"] = len(results) - invalid
    _push_metric("coal_to_gas_regions", len(results))
    _push_metric("coal_to_gas_invalid", invalid)
    return len(results)


with DAG(
    dag_id="coal_to_gas_switching",
    description="Compute regional coal-to-gas switching share",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(2),
    catchup=False,
    max_active_runs=1,
    tags=["gas", "coal", "analytics"],
) as dag:
    PythonOperator(
        task_id="compute_switching_metrics",
        python_callable=run_switching,
        provide_context=True,
    )
