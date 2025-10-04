"""Daily pipeline congestion model refresh and forecast generation."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

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
    "retry_delay": timedelta(minutes=30),
}

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000")
DEFAULT_PIPELINES = json.loads(
    os.getenv(
        "PIPELINE_CONGESTION_TARGETS",
        json.dumps(
            [
                {
                    "pipeline_id": "TCO_MAINLINE",
                    "market": "US_NE",
                    "segment": "mainline",
                    "flow_entity_id": "TCO_MAINLINE",
                    "flow_variable": "pipeline_flow_bcfd",
                    "weather_entity_id": "NYC",
                    "weather_variables": ["temperature"],
                    "demand_entity_id": "US_NE",
                    "demand_variable": "gas_demand_bcfd",
                }
            ]
        ),
    )
)


def _post(path: str, payload: dict) -> dict:
    url = f"{ML_SERVICE_URL}{path}"
    logger.info("Calling ML service: %s", url)
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def run_pipeline_congestion(**context) -> int:
    logical_date: datetime = context.get("logical_date") or datetime.utcnow()
    dag_conf = getattr(context.get("dag_run"), "conf", {})
    pipelines: List[Dict[str, str]] = dag_conf.get("pipelines") or DEFAULT_PIPELINES

    dq = DataQualityFramework()
    rules = dq.get_metric_rules("supply_chain")
    risk_upper = rules["pipeline_congestion_probability"]["value_max"]
    total = 0

    for pipeline in pipelines:
        payload = {
            "pipeline_id": pipeline["pipeline_id"],
            "market": pipeline.get("market"),
            "segment": pipeline.get("segment"),
            "flow_entity_id": pipeline["flow_entity_id"],
            "flow_variable": pipeline["flow_variable"],
            "weather_entity_id": pipeline.get("weather_entity_id"),
            "weather_variables": pipeline.get("weather_variables"),
            "demand_entity_id": pipeline.get("demand_entity_id"),
            "demand_variable": pipeline.get("demand_variable"),
            "horizon_days": int(pipeline.get("horizon_days", 7)),
        }
        response = _post("/api/v1/pipelines/congestion-train", payload)
        total += response.get("persisted_rows", 0)
        best_model = response.get("best_model")
        logger.info(
            "Pipeline %s trained using %s (persisted_rows=%s)",
            pipeline["pipeline_id"],
            best_model,
            response.get("persisted_rows", 0),
        )

        forecasts = requests.get(
            f"{ML_SERVICE_URL}/api/v1/pipelines/congestion-forecast",
            params={"pipeline_id": pipeline["pipeline_id"], "limit": pipeline.get("horizon_days", 7)},
            timeout=60,
        )
        forecasts.raise_for_status()
        forecast_payload = forecasts.json()
        for point in forecast_payload.get("forecasts", []):
            if point["congestion_probability"] > risk_upper:
                logger.warning(
                    "Forecast congestion probability %.2f exceeds %.2f for pipeline %s on %s",
                    point["congestion_probability"],
                    risk_upper,
                    pipeline["pipeline_id"],
                    point["forecast_date"],
                )
                total += 1
    return total


with DAG(
    dag_id="pipeline_congestion_forecast",
    description="Train and persist pipeline congestion forecasts",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 4 * * *",
    start_date=days_ago(3),
    catchup=False,
    max_active_runs=1,
    tags=["gas", "pipelines"],
) as dag:
    PythonOperator(
        task_id="train_pipeline_models",
        python_callable=run_pipeline_congestion,
        provide_context=True,
    )
