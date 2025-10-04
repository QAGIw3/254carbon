"""AGSI+ (GIE) Gas Storage Ingestion DAG."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


sys.path.append(os.path.join(os.path.dirname(__file__), "../../connectors"))
from external.infrastructure.agsi_connector import AGSIConnector  # noqa: E402


default_args = {
    "owner": "market-intelligence",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["ops@254carbon.ai"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _parse_json_list(val: str | None) -> list[str]:
    if not val:
        return []
    try:
        data = json.loads(val)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def run_agsi_ingestion(**context):
    api_base = os.getenv("AGSI_API_BASE", Variable.get("AGSI_API_BASE", default_var="https://agsi.gie.eu/api/v1"))
    api_key = os.getenv("AGSI_API_KEY", Variable.get("AGSI_API_KEY"))
    if not api_key:
        raise ValueError("AGSI_API_KEY is required")

    granularity = os.getenv("AGSI_GRANULARITY", Variable.get("AGSI_GRANULARITY", default_var="facility"))
    entities_csv = os.getenv("AGSI_ENTITIES", Variable.get("AGSI_ENTITIES", default_var=""))
    entities = [e.strip() for e in entities_csv.split(",") if e.strip()]
    include_rollups = os.getenv("AGSI_INCLUDE_ROLLUPS", Variable.get("AGSI_INCLUDE_ROLLUPS", default_var="false"))

    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", Variable.get("KAFKA_BOOTSTRAP", default_var="kafka:9092"))

    cfg = {
        "source_id": f"agsi_{granularity}",
        "api_base": api_base,
        "api_key": api_key,
        "granularity": granularity,
        "entities": entities or None,
        "include_rollups": include_rollups.lower() in {"1", "true", "yes"},
        "kafka_bootstrap": kafka_bootstrap,
        "kafka": {
            "topic": "market.fundamentals",
            "bootstrap_servers": kafka_bootstrap,
        },
    }

    connector = AGSIConnector(cfg)
    return connector.run()


with DAG(
    dag_id="agsi_ingestion",
    default_args=default_args,
    description="Ingest GIE AGSI+ gas storage metrics",
    schedule_interval=os.getenv("AGSI_SCHEDULE", Variable.get("AGSI_SCHEDULE", default_var="0 6 * * *")),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    ingest = PythonOperator(
        task_id="ingest_agsi",
        python_callable=run_agsi_ingestion,
        provide_context=True,
    )

