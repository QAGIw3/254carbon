"""
Eurostat Ingestion DAG

Runs EurostatConnector in live mode via SDMX JSON or bulk TSV downloads.
Backfill supports SDMX by chunking the 'time' parameter across year windows.
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
from external.demographics.eurostat_connector import EurostatConnector


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


def run_eurostat_live(**context):
    sdmx_queries = _get_json_list('EUROSTAT_SDMX_QUERIES')
    bulk_downloads = _get_json_list('EUROSTAT_BULK_DOWNLOADS')

    cfg = {
        'source_id': 'eurostat_live',
        'live': True,
        'sdmx_queries': sdmx_queries or None,
        'bulk_downloads': bulk_downloads or None,
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }
    connector = EurostatConnector(cfg)
    return connector.run()


def run_eurostat_backfill(**context):
    sdmx_queries = _get_json_list('EUROSTAT_SDMX_QUERIES')
    if not sdmx_queries:
        return 0

    start_year = int(os.getenv('EUROSTAT_BACKFILL_START_YEAR', Variable.get('EUROSTAT_BACKFILL_START_YEAR', default_var='2000')))
    end_year = int(os.getenv('EUROSTAT_BACKFILL_END_YEAR', Variable.get('EUROSTAT_BACKFILL_END_YEAR', default_var='2025')))
    step_years = int(os.getenv('EUROSTAT_BACKFILL_STEP_YEARS', Variable.get('EUROSTAT_BACKFILL_STEP_YEARS', default_var='5')))

    total = 0
    cur = start_year
    while cur <= end_year:
        chunk_end = min(cur + step_years - 1, end_year)
        queries = []
        for q in sdmx_queries:
            q2 = dict(q)
            params = dict(q2.get('params', {}))
            params['time'] = f"{cur}-{chunk_end}"
            q2['params'] = params
            queries.append(q2)

        cfg = {
            'source_id': f'eurostat_backfill_{cur}_{chunk_end}',
            'live': True,
            'sdmx_queries': queries,
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
        }
        connector = EurostatConnector(cfg)
        total += connector.run()
        cur = chunk_end + 1
    return total


with DAG(
    dag_id='eurostat_ingestion',
    default_args=default_args,
    description='Ingest Eurostat via SDMX or bulk TSV into Kafka fundamentals',
    schedule_interval=os.getenv('EUROSTAT_SCHEDULE', '@monthly'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    backfill = PythonOperator(
        task_id='backfill_eurostat',
        python_callable=run_eurostat_backfill,
        provide_context=True,
    )
    ingest = PythonOperator(
        task_id='ingest_eurostat',
        python_callable=run_eurostat_live,
        provide_context=True,
    )

    backfill >> ingest

