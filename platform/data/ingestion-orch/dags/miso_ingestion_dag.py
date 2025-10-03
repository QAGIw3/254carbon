"""
MISO Real-Time and Day-Ahead LMP Ingestion DAG
Scheduled to run every 5 minutes for RT, hourly for DA.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from miso_connector import MISOConnector

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_miso_rt_ingestion(**context):
    """Execute MISO real-time LMP ingestion."""
    config = {
        'source_id': 'miso_rt_lmp',
        'market_type': 'RT',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }
    
    connector = MISOConnector(config)
    events_processed = connector.run()
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    
    return events_processed


def run_miso_da_ingestion(**context):
    """Execute MISO day-ahead LMP ingestion."""
    config = {
        'source_id': 'miso_da_lmp',
        'market_type': 'DA',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }
    
    connector = MISOConnector(config)
    events_processed = connector.run()
    
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    
    return events_processed


def check_data_quality(**context):
    """Validate data quality of ingested data."""
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_miso_rt', key='events_processed')
    
    if events_processed < 100:
        raise ValueError(f"Too few events processed: {events_processed}")
    
    # Additional quality checks
    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    
    # Check for data freshness
    result = pg_hook.get_first("""
        SELECT MAX(event_time) as latest
        FROM ch.market_price_ticks
        WHERE source LIKE 'miso%'
    """)
    
    if result:
        latest_time = result[0]
        if latest_time and (datetime.utcnow() - latest_time).total_seconds() > 600:
            raise ValueError(f"Data staleness detected: latest={latest_time}")
    
    return True


# Real-time ingestion DAG (every 5 minutes)
with DAG(
    'miso_rt_ingestion',
    default_args=default_args,
    description='MISO real-time LMP data ingestion',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'miso', 'realtime'],
) as dag_rt:
    
    ingest_task = PythonOperator(
        task_id='ingest_miso_rt',
        python_callable=run_miso_rt_ingestion,
        provide_context=True,
    )
    
    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
        provide_context=True,
    )
    
    ingest_task >> quality_check


# Day-ahead ingestion DAG (hourly)
with DAG(
    'miso_da_ingestion',
    default_args=default_args,
    description='MISO day-ahead LMP data ingestion',
    schedule_interval='0 * * * *',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'miso', 'day-ahead'],
) as dag_da:
    
    ingest_task = PythonOperator(
        task_id='ingest_miso_da',
        python_callable=run_miso_da_ingestion,
        provide_context=True,
    )

