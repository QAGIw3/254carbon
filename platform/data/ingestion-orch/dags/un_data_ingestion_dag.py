"""
UN Data Ingestion DAG

Runs UNDataConnector in live mode via CSV downloads or SDMX queries. Backfill
supports SDMX by chunking the 'time' parameter across year windows.
"""
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))
from external.demographics.un_data_connector import UNDataConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def _get_json_list(name: str) -> list:
    raw = os.getenv(name, Variable.get(name, default_var=''))
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def run_un_live(**context):
    mode = os.getenv('UN_MODE', Variable.get('UN_MODE', default_var='csv')).lower()
    downloads = _get_json_list('UN_DOWNLOADS')
    sdmx_queries = _get_json_list('UN_SDMX_QUERIES')

    cfg = {
        'source_id': 'un_data_live',
        'live': True,
        'mode': mode,
        'downloads': downloads or None,
        'sdmx_queries': sdmx_queries or None,
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }
    connector = UNDataConnector(cfg)
    return connector.run()


def run_un_backfill(**context):
    sdmx_queries = _get_json_list('UN_SDMX_QUERIES')
    if not sdmx_queries:
        return 0

    start_year = int(os.getenv('UN_BACKFILL_START_YEAR', Variable.get('UN_BACKFILL_START_YEAR', default_var='2000')))
    end_year = int(os.getenv('UN_BACKFILL_END_YEAR', Variable.get('UN_BACKFILL_END_YEAR', default_var='2025')))
    step_years = int(os.getenv('UN_BACKFILL_STEP_YEARS', Variable.get('UN_BACKFILL_STEP_YEARS', default_var='5')))

    total = 0
    cur = start_year
    while cur <= end_year:
        chunk_end = min(cur + step_years - 1, end_year)
        queries = []
        for q in sdmx_queries:
            q2 = dict(q)
            q2['time_param'] = f"time={cur}-{chunk_end}"
            queries.append(q2)

        cfg = {
            'source_id': f'un_data_backfill_{cur}_{chunk_end}',
            'live': True,
            'mode': 'sdmx',
            'sdmx_queries': queries,
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
        }
        connector = UNDataConnector(cfg)
        total += connector.run()
        cur = chunk_end + 1
    return total


with DAG(
    dag_id='un_data_ingestion',
    default_args=default_args,
    description='Ingest UN Data (population/health/education) via CSV/SDMX into Kafka',
    schedule_interval=os.getenv('UN_SCHEDULE', '@monthly'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    backfill = PythonOperator(
        task_id='backfill_un_data',
        python_callable=run_un_backfill,
        provide_context=True,
    )
    ingest = PythonOperator(
        task_id='ingest_un_data',
        python_callable=run_un_live,
        provide_context=True,
    )

    backfill >> ingest

