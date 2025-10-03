"""
EPİAŞ (Turkey) Ingestion DAGs
Day-ahead market (DAM) and Intraday (IDM) aggregates
"""
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from middle_east.epias_connector import TurkeyEPIASConnector

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_epias_dam(**context):
    config = {
        'source_id': 'epias_dam',
        'market_type': 'DAM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
        # 'tgt_token': '{{ var.value.EPIAS_TGT | default("") }}',  # optional Airflow Variable
    }
    connector = TurkeyEPIASConnector(config)
    count = connector.run()
    context['task_instance'].xcom_push(key='events_processed', value=count)
    return count


def run_epias_idm(**context):
    config = {
        'source_id': 'epias_idm',
        'market_type': 'IDM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
        # 'tgt_token': '{{ var.value.EPIAS_TGT | default("") }}',
    }
    connector = TurkeyEPIASConnector(config)
    count = connector.run()
    context['task_instance'].xcom_push(key='events_processed', value=count)
    return count


def check_epias_quality(**context):
    ti = context['task_instance']
    dam = ti.xcom_pull(task_ids='ingest_epias_dam', key='events_processed') or 0
    idm = ti.xcom_pull(task_ids='ingest_epias_idm', key='events_processed') or 0
    total = dam + idm
    if total < 50:
        raise ValueError(f"Low EPİAŞ event count: {total}")
    return True


with DAG(
    'epias_dam_ingestion',
    default_args=default_args,
    description='EPİAŞ day-ahead MCP ingestion',
    schedule_interval='0 * * * *',  # Hourly
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'epias', 'dam'],
) as dag_epias_dam:
    ingest = PythonOperator(
        task_id='ingest_epias_dam',
        python_callable=run_epias_dam,
        provide_context=True,
    )
    quality = PythonOperator(
        task_id='check_epias_quality',
        python_callable=check_epias_quality,
        provide_context=True,
    )
    ingest >> quality


with DAG(
    'epias_idm_ingestion',
    default_args=default_args,
    description='EPİAŞ intraday aggregated price ingestion',
    schedule_interval='*/30 * * * *',  # Half-hourly
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'epias', 'idm'],
) as dag_epias_idm:
    ingest = PythonOperator(
        task_id='ingest_epias_idm',
        python_callable=run_epias_idm,
        provide_context=True,
    )
    quality = PythonOperator(
        task_id='check_epias_quality',
        python_callable=check_epias_quality,
        provide_context=True,
    )
    ingest >> quality

