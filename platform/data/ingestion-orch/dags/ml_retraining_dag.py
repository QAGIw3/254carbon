"""Automated ML retraining DAG for monitoring and model refresh."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing import Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["ml-ops@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}


def monitor_models(**context):
    """Check performance of all production models and emit alerts."""
    import requests

    ml_service = "http://ml-service:8006"
    response = requests.get(f"{ml_service}/api/v1/models/active")
    response.raise_for_status()
    active_models = response.json()

    degraded: Dict[str, Dict] = {}

    for instrument_id in active_models:
        perf_resp = requests.get(
            f"{ml_service}/api/v1/models/{instrument_id}/performance", params={"days": 30}
        )
        if perf_resp.status_code != 200:
            continue

        metrics = perf_resp.json()
        status = metrics.get("overall", {}).get("status")
        if status in {"degraded", "failed"}:
            degraded[instrument_id] = metrics

    context["task_instance"].xcom_push(key="degraded_models", value=degraded)
    return degraded


def retrain_models(**context):
    """Trigger retraining for degraded models."""
    import requests

    ti = context["task_instance"]
    degraded_models = ti.xcom_pull(task_ids="monitor_models", key="degraded_models") or {}

    ml_service = "http://ml-service:8006"
    retrain_results = {}

    for instrument_id in degraded_models.keys():
        resp = requests.post(
            f"{ml_service}/api/v1/models/{instrument_id}/retrain",
            json={"trigger": "automated_dag", "window_days": 120},
            timeout=120,
        )
        if resp.status_code == 200:
            retrain_results[instrument_id] = resp.json()
        else:
            retrain_results[instrument_id] = {"status": "failed", "code": resp.status_code}

    ti.xcom_push(key="retrain_results", value=retrain_results)
    return retrain_results


def evaluate_retrain(**context):
    """Validate retraining outcomes and update monitoring dashboards."""
    import requests

    ti = context["task_instance"]
    retrain_results = ti.xcom_pull(task_ids="retrain_models", key="retrain_results") or {}

    ml_service = "http://ml-service:8006"
    success_count = 0

    for instrument_id, payload in retrain_results.items():
        if payload.get("status") != "success":
            continue

        run_id = payload.get("model_version")
        verification_resp = requests.get(
            f"{ml_service}/api/v1/models/{instrument_id}/verification",
            params={"model_version": run_id},
        )
        if verification_resp.status_code == 200:
            success_count += 1

    try:
        requests.post(
            "http://prometheus:9091/metrics/job/ml_retraining",
            data=f"successful_retrainings {success_count}\n",
            timeout=5,
        )
    except Exception:
        pass

    return success_count


with DAG(
    "ml_model_retraining",
    default_args=default_args,
    description="Monitor and retrain ML models automatically",
    schedule_interval="0 */6 * * *",  # Every 6 hours
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "retraining", "monitoring"],
) as dag:

    monitor = PythonOperator(
        task_id="monitor_models",
        python_callable=monitor_models,
        provide_context=True,
    )

    retrain = PythonOperator(
        task_id="retrain_models",
        python_callable=retrain_models,
        provide_context=True,
    )

    evaluate = PythonOperator(
        task_id="evaluate_retrain",
        python_callable=evaluate_retrain,
        provide_context=True,
    )

    monitor >> retrain >> evaluate

