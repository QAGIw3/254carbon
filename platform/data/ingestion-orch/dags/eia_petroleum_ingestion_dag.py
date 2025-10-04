"""EIA Petroleum ingestion DAG."""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from eia_petroleum_connector import EIAPetroleumConnector


def run_eia_petroleum_ingestion(**context):
    connector = EIAPetroleumConnector(
        {
            "source_id": "eia_petroleum",
            "api_key": context["params"].get("api_key", os.getenv("EIA_API_KEY", "")),
        }
    )
    return connector.run()


def check_eia_data_quality(**context):
    from db import get_clickhouse_client

    ch = get_clickhouse_client()
    records = ch.execute(
        """
        SELECT count()
        FROM market_intelligence.market_price_ticks
        WHERE source = 'eia_petroleum'
          AND event_time >= now() - INTERVAL 2 DAY
        """
    )
    return records[0][0] > 0


def build_dag(paused: bool) -> DAG:
    return DAG(
        "eia_petroleum_ingestion",
        default_args={
            "owner": "market-intelligence",
            "depends_on_past": False,
            "email_on_failure": True,
            "email": ["ops@254carbon.ai"],
            "retries": 3,
            "retry_delay": timedelta(minutes=5),
        },
        description="Daily ingestion of EIA petroleum benchmarks",
        schedule_interval="0 6 * * *",
        start_date=days_ago(1),
        catchup=False,
        tags=["ingestion", "oil", "eia"],
        is_paused_upon_creation=paused,
    )


globals()["dag_eia_petroleum"] = build_dag(paused=False)

with globals()["dag_eia_petroleum"] as dag:
    ingest = PythonOperator(
        task_id="ingest_eia_petroleum",
        python_callable=run_eia_petroleum_ingestion,
        provide_context=True,
    )

    dq_check = PythonOperator(
        task_id="check_eia_data_quality",
        python_callable=check_eia_data_quality,
        provide_context=True,
    )

    ingest >> dq_check


