"""
US EIA Open Data Ingestion DAG

Runs EIA v2 API ingestion via EIAOpenDataConnector to publish indicators to
Kafka fundamentals. Supports simple dataset pulls or advanced query specs.
Includes optional backfill in month-chunk windows across a date range.
"""
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))
from external.infrastructure.eia_connector import EIAOpenDataConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def _parse_csv(val: str | None) -> list[str]:
    if not val:
        return []
    return [x.strip() for x in val.split(',') if x.strip()]


def _parse_json_list(val: str | None) -> list[dict]:
    if not val:
        return []
    try:
        data = json.loads(val)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def run_eia_live(**context):
    api_base = os.getenv('EIA_API_BASE', Variable.get('EIA_API_BASE', default_var='https://api.eia.gov/v2'))
    api_key = os.getenv('EIA_API_KEY', Variable.get('EIA_API_KEY', default_var=''))
    datasets = _parse_csv(os.getenv('EIA_DATASETS', Variable.get('EIA_DATASETS', default_var='')))
    queries = _parse_json_list(os.getenv('EIA_QUERIES', Variable.get('EIA_QUERIES', default_var='')))

    cfg = {
        'source_id': 'eia_open_data_live',
        'live': True,
        'api_base': api_base,
        'api_key': api_key,
        'datasets': datasets or None,
        'queries': queries or None,
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }
    connector = EIAOpenDataConnector(cfg)
    return connector.run()


def run_eia_backfill(**context):
    api_base = os.getenv('EIA_API_BASE', Variable.get('EIA_API_BASE', default_var='https://api.eia.gov/v2'))
    api_key = os.getenv('EIA_API_KEY', Variable.get('EIA_API_KEY', default_var=''))
    queries = _parse_json_list(os.getenv('EIA_QUERIES', Variable.get('EIA_QUERIES', default_var='')))
    if not queries:
        # Backfill only supported with query specs (provides facets/variables)
        return 0

    start = os.getenv('EIA_BACKFILL_START', Variable.get('EIA_BACKFILL_START', default_var=None))
    end = os.getenv('EIA_BACKFILL_END', Variable.get('EIA_BACKFILL_END', default_var=None))
    if not start or not end:
        return 0

    step_months = int(os.getenv('EIA_BACKFILL_STEP_MONTHS', Variable.get('EIA_BACKFILL_STEP_MONTHS', default_var='6')))

    # Helper to add months to YYYY-MM
    def add_months(ym: str, m: int) -> str:
        y, mm = map(int, ym.split('-'))
        y += (mm - 1 + m) // 12
        mm = (mm - 1 + m) % 12 + 1
        return f"{y:04d}-{mm:02d}"

    # Normalize YYYY-MM
    def norm(ym: str) -> str:
        return ym if len(ym) == 7 else f"{ym[:4]}-{ym[4:6]}"

    cur = norm(start)
    end_m = norm(end)
    total = 0
    while True:
        # chunk end inclusive
        chunk_end = add_months(cur, step_months - 1)
        if chunk_end > end_m:
            chunk_end = end_m
        for q in queries:
            q_chunk = q.copy()
            q_chunk['start'] = cur
            q_chunk['end'] = chunk_end
            cfg = {
                'source_id': f"eia_backfill_{cur}_{chunk_end}",
                'live': True,
                'api_base': api_base,
                'api_key': api_key,
                'queries': [q_chunk],
                'kafka_topic': 'market.fundamentals',
                'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
            }
            connector = EIAOpenDataConnector(cfg)
            total += connector.run()

        if chunk_end >= end_m:
            break
        # next chunk starts one month after current chunk_end
        cur = add_months(chunk_end, 1)

    return total


with DAG(
    dag_id='eia_ingestion',
    default_args=default_args,
    description='Ingest US EIA Open Data indicators to Kafka fundamentals',
    schedule_interval=os.getenv('EIA_SCHEDULE', '@weekly'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    backfill = PythonOperator(
        task_id='backfill_eia',
        python_callable=run_eia_backfill,
        provide_context=True,
    )
    ingest = PythonOperator(
        task_id='ingest_eia',
        python_callable=run_eia_live,
        provide_context=True,
    )

    backfill >> ingest

