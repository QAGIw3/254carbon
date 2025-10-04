"""
OECD Energy Statistics Ingestion DAG

Runs OECDEnergyStatsConnector in live mode to ingest energy balances and
household prices via CSV downloads or SDMX queries. Backfill supports SDMX
by chunking the 'time' parameter across year windows.
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
from external.infrastructure.oecd_energy_connector import OECDEnergyStatsConnector


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


def run_oecd_live(**context):
    mode = os.getenv('OECD_MODE', Variable.get('OECD_MODE', default_var='csv')).lower()
    downloads = _get_json_list('OECD_DOWNLOADS')
    sdmx_queries = _get_json_list('OECD_SDMX_QUERIES')

    cfg = {
        'source_id': 'oecd_energy_live',
        'live': True,
        'mode': mode,
        'downloads': downloads or None,
        'sdmx_queries': sdmx_queries or None,
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }
    connector = OECDEnergyStatsConnector(cfg)
    return connector.run()


def run_oecd_backfill(**context):
    # Backfill only for SDMX queries where we can adjust time range
    sdmx_queries = _get_json_list('OECD_SDMX_QUERIES')
    if not sdmx_queries:
        return 0

    start_year = int(os.getenv('OECD_BACKFILL_START_YEAR', Variable.get('OECD_BACKFILL_START_YEAR', default_var='2000')))
    end_year = int(os.getenv('OECD_BACKFILL_END_YEAR', Variable.get('OECD_BACKFILL_END_YEAR', default_var='2025')))
    step_years = int(os.getenv('OECD_BACKFILL_STEP_YEARS', Variable.get('OECD_BACKFILL_STEP_YEARS', default_var='5')))

    total = 0
    cur = start_year
    while cur <= end_year:
        chunk_end = min(cur + step_years - 1, end_year)
        queries = []
        for q in sdmx_queries:
            q2 = dict(q)
            # add/update time_param (e.g., time=2010-2015)
            q2['time_param'] = f"time={cur}-{chunk_end}"
            queries.append(q2)

        cfg = {
            'source_id': f'oecd_energy_backfill_{cur}_{chunk_end}',
            'live': True,
            'mode': 'sdmx',
            'sdmx_queries': queries,
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
        }
        connector = OECDEnergyStatsConnector(cfg)
        total += connector.run()
        cur = chunk_end + 1
    return total


with DAG(
    dag_id='oecd_energy_ingestion',
    default_args=default_args,
    description='Ingest OECD Energy Statistics via CSV or SDMX into Kafka fundamentals',
    schedule_interval=os.getenv('OECD_SCHEDULE', '@monthly'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    backfill = PythonOperator(
        task_id='backfill_oecd_energy',
        python_callable=run_oecd_backfill,
        provide_context=True,
    )
    ingest = PythonOperator(
        task_id='ingest_oecd_energy',
        python_callable=run_oecd_live,
        provide_context=True,
    )

    backfill >> ingest

