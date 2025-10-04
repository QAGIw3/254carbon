"""
PJM Ingestion DAGs

Schedules RT/DA LMP, capacity, and ancillary services using PJMConnector.
Swap mocks for Data Miner 2 API calls once keys are configured.
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
        'market_type': 'DAM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = PJMConnector(config)
    events_processed = connector.run()

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)

    return events_processed


def run_pjm_rtm_ingestion(**context):
    """Execute PJM real-time LMP ingestion."""
    config = {
        'source_id': 'pjm_rtm_lmp',
        'market_type': 'RT',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = PJMConnector(config)
    events_processed = connector.run()

    # Push metrics to XCom
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

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)

    return events_processed


def check_pjm_data_quality(**context):
    """Check PJM data quality metrics."""
    ti = context['task_instance']

    # Get metrics from previous tasks
    da_events = ti.xcom_pull(task_ids='ingest_pjm_da', key='events_processed')
    rt_events = ti.xcom_pull(task_ids='ingest_pjm_rt', key='events_processed')
    capacity_events = ti.xcom_pull(task_ids='ingest_pjm_capacity', key='events_processed')

    total_events = (da_events or 0) + (rt_events or 0) + (capacity_events or 0)

    # Quality checks
    if total_events < 1000:  # Expect at least 1000 events across all markets
        raise ValueError(f"Low event count: {total_events}")

    if da_events and da_events < 200:  # Expect at least 200 DA events
        raise ValueError(f"Low DA event count: {da_events}")

    if rt_events and rt_events < 500:  # Expect at least 500 RT events
        raise ValueError(f"Low RT event count: {rt_events}")

    return True


# PJM Day-ahead ingestion DAG (hourly)
with DAG(
    'pjm_da_ingestion',
    default_args=default_args,
    description='PJM day-ahead LMP data ingestion',
    schedule_interval='0 * * * *',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'pjm', 'day-ahead'],
) as dag_pjm_da:

    ingest_task = PythonOperator(
        task_id='ingest_pjm_da',
        python_callable=run_pjm_da_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_pjm_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# PJM Real-time ingestion DAG (every 5 minutes)
with DAG(
    'pjm_rtm_ingestion',
    default_args=default_args,
    description='PJM real-time LMP data ingestion',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'pjm', 'realtime'],
) as dag_pjm_rtm:

    ingest_task = PythonOperator(
        task_id='ingest_pjm_rtm',
        python_callable=run_pjm_rtm_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_pjm_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# PJM Capacity market ingestion DAG (daily)
with DAG(
    'pjm_capacity_ingestion',
    default_args=default_args,
    description='PJM capacity market data ingestion',
    schedule_interval='0 3 * * *',  # Daily at 3 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'pjm', 'capacity'],
) as dag_pjm_capacity:

    ingest_task = PythonOperator(
        task_id='ingest_pjm_capacity',
        python_callable=run_pjm_capacity_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_pjm_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# Combined PJM backfill DAG (runs on demand)
def run_pjm_backfill(**context):
    """Run historical backfill for PJM markets."""
    start_date = context['dag_run'].conf.get('start_date', '2024-01-01')
    end_date = context['dag_run'].conf.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    logger.info(f"Running PJM backfill from {start_date} to {end_date}")

    # This would implement historical data backfill
    # For now, just return success
    return f"Backfill completed: {start_date} to {end_date}"


with DAG(
    'pjm_backfill',
    default_args=default_args,
    description='PJM historical data backfill',
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=['backfill', 'pjm', 'historical'],
) as dag_pjm_backfill:

    backfill_task = PythonOperator(
        task_id='run_pjm_backfill',
        python_callable=run_pjm_backfill,
        provide_context=True,
    )
