"""Coal stockpile monitoring DAG."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import List

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


sys.path.append(os.path.join(os.path.dirname(__file__), "../../connectors"))
from external.infrastructure.coal_satellite_connector import (  # noqa: E402
    CoalSatelliteConnector,
    load_sites_from_csv,
)


default_args = {
    "owner": "market-intelligence",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["ops@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def run_coal_stockpile_ingestion(**context):
    api_base = os.getenv(
        "SATELLITE_INTEL_BASE",
        Variable.get("SATELLITE_INTEL_BASE", default_var="http://satellite-intel:8025"),
    )
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", Variable.get("KAFKA_BOOTSTRAP", default_var="kafka:9092"))

    sites_csv_env = os.getenv("COAL_SITES_CSV")
    base_path = Path("/home/m/254carbon/platform/data/reference/coal_sites.csv")
    sites_path = Path(sites_csv_env) if sites_csv_env else base_path

    sites = load_sites_from_csv(str(sites_path))

    cfg = {
        "source_id": "coal_satellite",
        "api_base": api_base,
        "sites": [site.__dict__ for site in sites],
        "kafka": {
            "topic": "market.fundamentals",
            "bootstrap_servers": kafka_bootstrap,
        },
    }

    connector = CoalSatelliteConnector(cfg)
    return connector.run()


with DAG(
    dag_id="coal_stockpile_monitoring",
    default_args=default_args,
    description="Monitor coal stockpiles via satellite-intel",
    schedule_interval=os.getenv(
        "COAL_STOCKPILE_SCHEDULE",
        Variable.get("COAL_STOCKPILE_SCHEDULE", default_var="0 5 * * MON"),
    ),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    ingest = PythonOperator(
        task_id="ingest_coal_stockpiles",
        python_callable=run_coal_stockpile_ingestion,
        provide_context=True,
    )

