"""
Open Infrastructure Map (Overpass) Ingestion DAG

Runs OpenInfrastructureMapConnector in live mode to summarize power lines,
substations, and pipelines for a configured bbox or area. Scheduled snapshot.
"""
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))
from external.infrastructure.open_inframap_connector import OpenInfrastructureMapConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def _get_str(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, Variable.get(name, default_var=default))
    return val


def _get_bbox(name: str) -> list[float] | None:
    val = _get_str(name, None)
    if not val:
        return None
    try:
        parts = [float(x.strip()) for x in val.split(',')]
        if len(parts) == 4:
            return parts
    except Exception:
        return None
    return None


def run_oim_snapshot(**context):
    cfg = {
        'source_id': 'open_inframap_live',
        'live': True,
        'overpass_url': _get_str('OIM_OVERPASS_URL', 'https://overpass-api.de/api/interpreter'),
        'bbox': _get_bbox('OIM_BBOX'),
        'area_id': (int(_get_str('OIM_AREA_ID')) if _get_str('OIM_AREA_ID') else None),
        'region_name': _get_str('OIM_REGION_NAME', 'WORLD') or 'WORLD',
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }

    connector = OpenInfrastructureMapConnector(cfg)
    return connector.run()


with DAG(
    dag_id='open_inframap_ingestion',
    default_args=default_args,
    description='Summarize OIM infrastructure via Overpass into Kafka fundamentals',
    schedule_interval=os.getenv('OIM_SCHEDULE', '@weekly'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    ingest = PythonOperator(
        task_id='ingest_oim_snapshot',
        python_callable=run_oim_snapshot,
        provide_context=True,
    )

