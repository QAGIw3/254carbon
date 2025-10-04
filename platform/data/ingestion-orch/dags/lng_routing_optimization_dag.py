"""Daily LNG routing optimisation orchestration."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from platform.shared.data_quality_framework import DataQualityFramework

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "supply-chain",
    "email_on_failure": True,
    "email": ["analytics@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000")
DEFAULT_EXPORT_TERMINALS = os.getenv(
    "LNG_EXPORT_TERMINALS",
    "Sabine Pass,Corpus Christi,Ras Laffan",
).split(",")
DEFAULT_IMPORT_TERMINALS = os.getenv(
    "LNG_IMPORT_TERMINALS",
    "Zeebrugge,Rotterdam,Tokyo",
).split(",")


def _post(path: str, payload: dict) -> dict:
    url = f"{ML_SERVICE_URL}{path}"
    logger.info("Calling ML service: %s", url)
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _terminals_from_conf(conf: dict | None, key: str, default: List[str]) -> List[str]:
    if conf and conf.get(key):
        value = conf[key]
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
    return [item.strip() for item in default if item.strip()]


def run_lng_routing(**context) -> int:
    logical_date: datetime = context.get("logical_date") or datetime.utcnow()
    dag_conf = getattr(context.get("dag_run"), "conf", {})
    export_terminals = _terminals_from_conf(dag_conf, "export_terminals", DEFAULT_EXPORT_TERMINALS)
    import_terminals = _terminals_from_conf(dag_conf, "import_terminals", DEFAULT_IMPORT_TERMINALS)

    payload = {
        "as_of_date": logical_date.date().isoformat(),
        "export_terminals": export_terminals,
        "import_terminals": import_terminals,
        "cargo_size_bcf": float(dag_conf.get("cargo_size_bcf", 3.5)),
        "vessel_speed_knots": float(dag_conf.get("vessel_speed_knots", 19.5)),
        "fuel_price_usd_per_tonne": float(dag_conf.get("fuel_price_usd_per_tonne", 600.0)),
    }

    result = _post("/api/v1/supplychain/lng/optimize-routes", payload)
    options = result.get("all_routes", [])
    if not options:
        logger.warning("ML service returned no routing options")
        return 0

    frame = pd.DataFrame(options)
    dq = DataQualityFramework()
    rules = dq.get_metric_rules("supply_chain")

    lower = rules["lng_cost_per_mmbtu_usd"]["value_min"]
    upper = rules["lng_cost_per_mmbtu_usd"]["value_max"]
    breached = frame[(frame["cost_per_mmbtu_usd"] < lower) | (frame["cost_per_mmbtu_usd"] > upper)]
    if not breached.empty:
        logger.warning("Detected %s routing cost anomalies", len(breached))
        for _, row in breached.iterrows():
            logger.warning(
                "Route %s->%s cost %.2f USD/MMBtu out of bounds",
                row["export_terminal"],
                row["import_terminal"],
                row["cost_per_mmbtu_usd"],
            )

    logger.info("Persisted %s LNG routing rows", result.get("persisted_rows", 0))
    return len(options)


with DAG(
    dag_id="lng_routing_optimization",
    description="Optimise LNG routing and persist analytics",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(2),
    catchup=False,
    max_active_runs=1,
    tags=["lng", "supply-chain"],
) as dag:
    PythonOperator(
        task_id="optimise_lng_routes",
        python_callable=run_lng_routing,
        provide_context=True,
    )
