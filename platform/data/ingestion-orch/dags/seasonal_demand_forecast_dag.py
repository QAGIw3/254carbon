"""Daily seasonal demand forecasting orchestration."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

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
    "retry_delay": timedelta(minutes=20),
}

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000")
DEFAULT_DEMAND_TARGETS = json.loads(
    os.getenv(
        "SEASONAL_DEMAND_TARGETS",
        json.dumps(
            [
                {
                    "region": "northeast",
                    "historical_entity_id": "US_NE",
                    "historical_variable": "gas_demand_bcfd",
                    "weather_entity_id": "NYC",
                    "weather_variable": "temperature",
                    "scenario_id": "BASE",
                }
            ]
        ),
    )
)


def _post(path: str, payload: dict) -> dict:
    url = f"{ML_SERVICE_URL}{path}"
    logger.info("Calling ML service: %s", url)
    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def run_seasonal_demand(**context) -> int:
    logical_date: datetime = context.get("logical_date") or datetime.utcnow()
    dag_conf = getattr(context.get("dag_run"), "conf", {})
    targets: List[Dict[str, str]] = dag_conf.get("targets") or DEFAULT_DEMAND_TARGETS
    horizon_days = int(dag_conf.get("horizon_days", 45))

    dq = DataQualityFramework()
    rules = dq.get_metric_rules("supply_chain")
    lower = rules["seasonal_demand_forecast_mw"]["value_min"]
    upper = rules["seasonal_demand_forecast_mw"]["value_max"]

    total_records = 0

    for target in targets:
        payload = {
            "region": target["region"],
            "scenario_id": target.get("scenario_id", "BASE"),
            "historical_entity_id": target["historical_entity_id"],
            "historical_variable": target["historical_variable"],
            "weather_entity_id": target["weather_entity_id"],
            "weather_variable": target["weather_variable"],
            "horizon_days": horizon_days,
            "economic_indicators": target.get("economic_indicators"),
        }
        response = _post("/api/v1/demand/seasonal-forecast", payload)
        records = response.get("records", [])
        total_records += len(records)
        frame = pd.DataFrame(records)
        if not frame.empty:
            out_of_bounds = frame[
                (frame["final_forecast_mw"] < lower)
                | (frame["final_forecast_mw"] > upper)
            ]
            if not out_of_bounds.empty:
                logger.warning(
                    "Seasonal demand forecast outliers detected for %s (%s rows)",
                    target["region"],
                    len(out_of_bounds),
                )
    return total_records


with DAG(
    dag_id="seasonal_demand_forecast",
    description="Generate seasonal demand forecasts with rolling weather inputs",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 5 * * *",
    start_date=days_ago(3),
    catchup=False,
    max_active_runs=1,
    tags=["demand", "gas"],
) as dag:
    PythonOperator(
        task_id="compute_seasonal_demand",
        python_callable=run_seasonal_demand,
        provide_context=True,
    )
