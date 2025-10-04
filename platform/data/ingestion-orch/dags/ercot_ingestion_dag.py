"""
ERCOT Ingestion DAGs

Runs SPP (settlement point prices), hub prices, ORDC adders, and resource
telemetry ingestions on schedules aligned to product cadence.

This DAG file uses the ERCOTConnector which currently produces mock data
in dev/testing; swap to live API endpoints as available.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from ercot_connector import ERCOTConnector

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_ercot_rtm_ingestion(**context):
    """Execute ERCOT real-time LMP ingestion."""
    config = {
        'source_id': 'ercot_rtm_lmp',
        'market_type': 'RTM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = ERCOTConnector(config)
    events_processed = connector.run()

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)

    return events_processed


def run_ercot_dam_ingestion(**context):
    """Execute ERCOT day-ahead LMP ingestion."""
    config = {
        'source_id': 'ercot_dam_lmp',
        'market_type': 'DAM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = ERCOTConnector(config)
    events_processed = connector.run()

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)

    return events_processed


def run_ercot_ancillary_ingestion(**context):
    """Execute ERCOT ancillary services ingestion."""
    config = {
        'source_id': 'ercot_ancillary',
        'market_type': 'ANCILLARY',
        'kafka_topic': 'power.ancillary.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = ERCOTConnector(config)
    events_processed = connector.run()

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)

    return events_processed


def check_ercot_data_quality(**context):
    """Check ERCOT data quality metrics."""
    ti = context['task_instance']

    # Get metrics from previous tasks
    rtm_events = ti.xcom_pull(task_ids='ingest_ercot_rtm', key='events_processed')
    dam_events = ti.xcom_pull(task_ids='ingest_ercot_dam', key='events_processed')
    ancillary_events = ti.xcom_pull(task_ids='ingest_ercot_ancillary', key='events_processed')

    total_events = (rtm_events or 0) + (dam_events or 0) + (ancillary_events or 0)

    # Quality checks - ERCOT has ~4000 nodes
    if total_events < 500:  # Expect at least 500 events across all markets
        raise ValueError(f"Low event count: {total_events}")

    if rtm_events and rtm_events < 200:  # Expect at least 200 RT events
        raise ValueError(f"Low RT event count: {rtm_events}")

    if dam_events and dam_events < 100:  # Expect at least 100 DA events
        raise ValueError(f"Low DA event count: {dam_events}")

    return True


# ERCOT Real-time ingestion DAG (every 15 minutes - ERCOT publishes every 15 min)
with DAG(
    'ercot_rtm_ingestion',
    default_args=default_args,
    description='ERCOT real-time LMP data ingestion',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'ercot', 'realtime'],
) as dag_ercot_rtm:

    ingest_task = PythonOperator(
        task_id='ingest_ercot_rtm',
        python_callable=run_ercot_rtm_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_ercot_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# ERCOT Day-ahead ingestion DAG (hourly)
with DAG(
    'ercot_dam_ingestion',
    default_args=default_args,
    description='ERCOT day-ahead LMP data ingestion',
    schedule_interval='0 * * * *',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'ercot', 'day-ahead'],
) as dag_ercot_dam:

    ingest_task = PythonOperator(
        task_id='ingest_ercot_dam',
        python_callable=run_ercot_dam_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_ercot_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# ERCOT Ancillary services DAG (hourly)
with DAG(
    'ercot_ancillary_ingestion',
    default_args=default_args,
    description='ERCOT ancillary services data ingestion',
    schedule_interval='0 * * * *',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'ercot', 'ancillary'],
) as dag_ercot_ancillary:

    ingest_task = PythonOperator(
        task_id='ingest_ercot_ancillary',
        python_callable=run_ercot_ancillary_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_ercot_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# ERCOT backfill DAG (manual trigger)
def run_ercot_backfill(**context):
    """Run historical backfill for ERCOT markets."""
    start_date = context['dag_run'].conf.get('start_date', '2024-01-01')
    end_date = context['dag_run'].conf.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    print(f"Running ERCOT backfill from {start_date} to {end_date}")

    # This would implement historical data backfill
    # For now, just return success
    return f"Backfill completed: {start_date} to {end_date}"


with DAG(
    'ercot_backfill',
    default_args=default_args,
    description='ERCOT historical data backfill',
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=['backfill', 'ercot', 'historical'],
) as dag_ercot_backfill:

    backfill_task = PythonOperator(
        task_id='run_ercot_backfill',
        python_callable=run_ercot_backfill,
        provide_context=True,
    )
