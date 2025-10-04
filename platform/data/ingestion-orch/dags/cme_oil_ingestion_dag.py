"""CME oil futures ingestion DAG (stub)."""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from cme_oil_connector import CMEGroupConnector


def run_cme_oil_ingestion(**context):
    connector = CMEGroupConnector(
        {
            "source_id": "cme_oil_futures",
            "api_base_url": context["params"].get("api_base_url"),
            "api_key": context["params"].get("api_key"),
        }
    )
    return connector.run()


dag = DAG(
    "cme_oil_ingestion",
    default_args={
        "owner": "market-intelligence",
        "depends_on_past": False,
        "email_on_failure": True,
        "email": ["ops@254carbon.ai"],
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    description="CME oil futures ingestion (paused until credentials provided)",
    schedule_interval="0 7 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["ingestion", "oil", "cme"],
    is_paused_upon_creation=True,
)

with dag:
    PythonOperator(
        task_id="ingest_cme_oil_futures",
        python_callable=run_cme_oil_ingestion,
        provide_context=True,
    )
