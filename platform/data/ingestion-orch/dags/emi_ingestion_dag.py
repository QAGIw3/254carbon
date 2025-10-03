"""
New Zealand EMI Ingestion DAGs
Half-hourly final price (island-level)
"""
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from apac.new_zealand_emi_connector import NewZealandEMIConnector

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_emi_finalprice_ni(**context):
    config = {
        'source_id': 'emi_finalprice_ni',
        'series': 'final_price',
        'scope': 'island',
        'island': 'NI',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
        # 'api_key': '{{ var.value.EMI_API_KEY | default("") }}',
    }
    connector = NewZealandEMIConnector(config)
    count = connector.run()
    context['task_instance'].xcom_push(key='events_processed', value=count)
    return count


def run_emi_finalprice_si(**context):
    config = {
        'source_id': 'emi_finalprice_si',
        'series': 'final_price',
        'scope': 'island',
        'island': 'SI',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
        # 'api_key': '{{ var.value.EMI_API_KEY | default("") }}',
    }
    connector = NewZealandEMIConnector(config)
    count = connector.run()
    context['task_instance'].xcom_push(key='events_processed', value=count)
    return count


def check_emi_quality(**context):
    ti = context['task_instance']
    ni = ti.xcom_pull(task_ids='ingest_emi_finalprice_ni', key='events_processed') or 0
    si = ti.xcom_pull(task_ids='ingest_emi_finalprice_si', key='events_processed') or 0
    total = ni + si
    if total < 40:
        raise ValueError(f"Low EMI event count: {total}")
    return True


with DAG(
    'emi_finalprice_ingestion',
    default_args=default_args,
    description='EMI final price ingestion (island-level)',
    schedule_interval='*/30 * * * *',  # Half-hourly
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'emi', 'finalprice'],
) as dag_emi_final:
    ingest_ni = PythonOperator(
        task_id='ingest_emi_finalprice_ni',
        python_callable=run_emi_finalprice_ni,
        provide_context=True,
    )
    ingest_si = PythonOperator(
        task_id='ingest_emi_finalprice_si',
        python_callable=run_emi_finalprice_si,
        provide_context=True,
    )
    quality = PythonOperator(
        task_id='check_emi_quality',
        python_callable=check_emi_quality,
        provide_context=True,
    )
    [ingest_ni, ingest_si] >> quality

