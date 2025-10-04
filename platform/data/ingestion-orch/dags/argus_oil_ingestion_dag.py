"""Argus oil assessments ingestion DAG (stub)."""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from argus_connector import ArgusOilConnector


def run_argus_oil_ingestion(**context):
    connector = ArgusOilConnector(
        {
            "source_id": "argus_oil_assessments",
            "market": context["params"].get("market", "ARGUS"),
        }
    )
    return connector.run()


dag = DAG(
    "argus_oil_ingestion",
    default_args={
        "owner": "market-intelligence",
        "depends_on_past": False,
        "email_on_failure": True,
        "email": ["ops@254carbon.ai"],
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    description="Argus oil assessments ingestion (paused until credentials provided)",
    schedule_interval="0 9 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["ingestion", "oil", "argus"],
    is_paused_upon_creation=True,
)

with dag:
    PythonOperator(
        task_id="ingest_argus_oil_assessments",
        python_callable=run_argus_oil_ingestion,
        provide_context=True,
    )
