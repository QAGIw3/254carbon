"""
PJM Data Ingestion DAGs
Real-time, day-ahead, capacity, and ancillary services.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from pjm_connector import PJMConnector

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_pjm_rt_ingestion(**context):
    """Execute PJM real-time LMP ingestion."""
    config = {
        'source_id': 'pjm_rt_lmp',
        'market_type': 'RT',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }
    
    connector = PJMConnector(config)
    events_processed = connector.run()
    
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_pjm_da_ingestion(**context):
    """Execute PJM day-ahead LMP ingestion."""
    config = {
        'source_id': 'pjm_da_lmp',
        'market_type': 'DA',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }
    
    connector = PJMConnector(config)
    events_processed = connector.run()
    
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_pjm_capacity_ingestion(**context):
    """Execute PJM capacity market ingestion."""
    config = {
        'source_id': 'pjm_capacity',
        'market_type': 'CAPACITY',
        'kafka_topic': 'power.capacity.v1',
        'kafka_bootstrap': 'kafka:9092',
    }
    
    connector = PJMConnector(config)
    events_processed = connector.run()
    
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


# Real-time LMP DAG (every 5 minutes)
with DAG(
    'pjm_rt_ingestion',
    default_args=default_args,
    description='PJM real-time LMP data ingestion',
    schedule_interval='*/5 * * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'pjm', 'realtime'],
) as dag_rt:
    
    ingest_task = PythonOperator(
        task_id='ingest_pjm_rt',
        python_callable=run_pjm_rt_ingestion,
        provide_context=True,
    )


# Day-ahead LMP DAG (hourly)
with DAG(
    'pjm_da_ingestion',
    default_args=default_args,
    description='PJM day-ahead LMP data ingestion',
    schedule_interval='0 * * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'pjm', 'day-ahead'],
) as dag_da:
    
    ingest_task = PythonOperator(
        task_id='ingest_pjm_da',
        python_callable=run_pjm_da_ingestion,
        provide_context=True,
    )


# Capacity market DAG (daily)
with DAG(
    'pjm_capacity_ingestion',
    default_args=default_args,
    description='PJM capacity market data ingestion',
    schedule_interval='0 8 * * *',  # 8 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'pjm', 'capacity'],
) as dag_capacity:
    
    ingest_task = PythonOperator(
        task_id='ingest_pjm_capacity',
        python_callable=run_pjm_capacity_ingestion,
        provide_context=True,
    )

