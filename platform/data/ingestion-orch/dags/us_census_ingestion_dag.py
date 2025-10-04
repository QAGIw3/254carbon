"""
US Census Ingestion DAG

Runs USCensusConnector in live mode for one or more query specs defined via
Airflow Variable/ENV `CENSUS_QUERIES` (JSON array). Supports simple backfill
across years when the dataset string contains a `{year}` placeholder (e.g.,
"{year}/acs/acs1").
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
from external.demographics.us_census_connector import USCensusConnector


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


def run_census_live(**context):
    api_base = os.getenv('CENSUS_API_BASE', Variable.get('CENSUS_API_BASE', default_var='https://api.census.gov/data'))
    api_key = os.getenv('CENSUS_API_KEY', Variable.get('CENSUS_API_KEY', default_var=''))
    specs = _get_json_list('CENSUS_QUERIES')
    if not specs:
        return 0

    total = 0
    for spec in specs:
        cfg = {
            'source_id': f"census_{spec.get('dataset','dataset').replace('/','_')}",
            'live': True,
            'api_base': api_base,
            'api_key': api_key or spec.get('api_key'),
            'dataset': spec.get('dataset', '2020/dec/pl'),
            'variables': spec.get('variables', ['NAME','P1_001N']),
            'geo_for': spec.get('geo_for', 'state:*'),
            'geo_in': spec.get('geo_in'),
            'timestamp_year_field': spec.get('timestamp_year_field'),
            'year': spec.get('year'),
            'aliases': spec.get('aliases'),
            'entity_field': spec.get('entity_field', 'NAME'),
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
        }
        connector = USCensusConnector(cfg)
        total += connector.run()
    return total


def run_census_backfill(**context):
    api_base = os.getenv('CENSUS_API_BASE', Variable.get('CENSUS_API_BASE', default_var='https://api.census.gov/data'))
    api_key = os.getenv('CENSUS_API_KEY', Variable.get('CENSUS_API_KEY', default_var=''))
    specs = _get_json_list('CENSUS_QUERIES')
    if not specs:
        return 0

    start_year = int(os.getenv('CENSUS_BACKFILL_START_YEAR', Variable.get('CENSUS_BACKFILL_START_YEAR', default_var='2010')))
    end_year = int(os.getenv('CENSUS_BACKFILL_END_YEAR', Variable.get('CENSUS_BACKFILL_END_YEAR', default_var='2025')))
    total = 0

    for y in range(start_year, end_year + 1):
        for spec in specs:
            dataset = spec.get('dataset', '2020/dec/pl')
            # If dataset includes a year placeholder, substitute; else use as-is and set year field if provided
            ds = dataset.format(year=y) if '{year}' in dataset else dataset
            cfg = {
                'source_id': f"census_{ds.replace('/','_')}_{y}",
                'live': True,
                'api_base': api_base,
                'api_key': api_key or spec.get('api_key'),
                'dataset': ds,
                'variables': spec.get('variables', ['NAME','P1_001N']),
                'geo_for': spec.get('geo_for', 'state:*'),
                'geo_in': spec.get('geo_in'),
                'timestamp_year_field': spec.get('timestamp_year_field'),
                'year': y,
                'aliases': spec.get('aliases'),
                'entity_field': spec.get('entity_field', 'NAME'),
                'kafka_topic': 'market.fundamentals',
                'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
            }
            connector = USCensusConnector(cfg)
            total += connector.run()
    return total


with DAG(
    dag_id='us_census_ingestion',
    default_args=default_args,
    description='Ingest US Census (population, housing) to Kafka fundamentals',
    schedule_interval=os.getenv('CENSUS_SCHEDULE', '@monthly'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    backfill = PythonOperator(
        task_id='backfill_census',
        python_callable=run_census_backfill,
        provide_context=True,
    )
    ingest = PythonOperator(
        task_id='ingest_census',
        python_callable=run_census_live,
        provide_context=True,
    )

    backfill >> ingest

