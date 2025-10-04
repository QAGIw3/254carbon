"""Platts oil assessments ingestion DAG (stub)."""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from platts_oil_connector import PlattsOilConnector


def run_platts_oil_ingestion(**context):
    connector = PlattsOilConnector(
        {
            "source_id": "platts_oil_assessments",
            "api_base_url": context["params"].get("api_base_url"),
            "api_key": context["params"].get("api_key"),
        }
    )
    return connector.run()


dag = DAG(
    "platts_oil_ingestion",
    default_args={
        "owner": "market-intelligence",
        "depends_on_past": False,
        "email_on_failure": True,
        "email": ["ops@254carbon.ai"],
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    description="Platts oil assessments ingestion (paused until credentials provided)",
    schedule_interval="0 8 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["ingestion", "oil", "platts"],
    is_paused_upon_creation=True,
)

with dag:
    PythonOperator(
        task_id="ingest_platts_oil_assessments",
        python_callable=run_platts_oil_ingestion,
        provide_context=True,
    )
