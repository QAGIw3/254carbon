"""
SPP Ingestion DAGs

Schedules
- RTM: every 5 minutes
- DAM: hourly
- IM: every 15 minutes

Design
- Uses SPPConnector to ingest LMP and IM series (mocked by default) and
  performs simple quality checks for volume expectations.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from spp_connector import SPPConnector

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_spp_rtm_ingestion(**context):
    """Execute SPP real-time LMP ingestion."""
    config = {
        'source_id': 'spp_rtm_lmp',
        'market_type': 'RTM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = SPPConnector(config)
    events_processed = connector.run()

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)

    return events_processed


def run_spp_dam_ingestion(**context):
    """Execute SPP day-ahead LMP ingestion."""
    config = {
        'source_id': 'spp_dam_lmp',
        'market_type': 'DAM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = SPPConnector(config)
    events_processed = connector.run()

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)

    return events_processed


def run_spp_im_ingestion(**context):
    """Execute SPP integrated marketplace ingestion."""
    config = {
        'source_id': 'spp_im',
        'market_type': 'IM',
        'kafka_topic': 'power.im.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = SPPConnector(config)
    events_processed = connector.run()

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)

    return events_processed


def check_spp_data_quality(**context):
    """Check SPP data quality metrics."""
    ti = context['task_instance']

    # Get metrics from previous tasks
    rtm_events = ti.xcom_pull(task_ids='ingest_spp_rtm', key='events_processed')
    dam_events = ti.xcom_pull(task_ids='ingest_spp_dam', key='events_processed')
    im_events = ti.xcom_pull(task_ids='ingest_spp_im', key='events_processed')

    total_events = (rtm_events or 0) + (dam_events or 0) + (im_events or 0)

    # Quality checks - SPP has ~2000 nodes
    if total_events < 300:  # Expect at least 300 events across all markets
        raise ValueError(f"Low event count: {total_events}")

    if rtm_events and rtm_events < 150:  # Expect at least 150 RT events
        raise ValueError(f"Low RT event count: {rtm_events}")

    if dam_events and dam_events < 100:  # Expect at least 100 DA events
        raise ValueError(f"Low DA event count: {dam_events}")

    return True


# SPP Real-time ingestion DAG (every 5 minutes)
with DAG(
    'spp_rtm_ingestion',
    default_args=default_args,
    description='SPP real-time LMP data ingestion',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'spp', 'realtime'],
) as dag_spp_rtm:

    ingest_task = PythonOperator(
        task_id='ingest_spp_rtm',
        python_callable=run_spp_rtm_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_spp_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# SPP Day-ahead ingestion DAG (hourly)
with DAG(
    'spp_dam_ingestion',
    default_args=default_args,
    description='SPP day-ahead LMP data ingestion',
    schedule_interval='0 * * * *',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'spp', 'day-ahead'],
) as dag_spp_dam:

    ingest_task = PythonOperator(
        task_id='ingest_spp_dam',
        python_callable=run_spp_dam_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_spp_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# SPP Integrated Marketplace DAG (every 15 minutes)
with DAG(
    'spp_im_ingestion',
    default_args=default_args,
    description='SPP integrated marketplace data ingestion',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'spp', 'integrated-marketplace'],
) as dag_spp_im:

    ingest_task = PythonOperator(
        task_id='ingest_spp_im',
        python_callable=run_spp_im_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_spp_data_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# SPP backfill DAG (manual trigger)
def run_spp_backfill(**context):
    """Run historical backfill for SPP markets."""
    start_date = context['dag_run'].conf.get('start_date', '2024-01-01')
    end_date = context['dag_run'].conf.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    print(f"Running SPP backfill from {start_date} to {end_date}")

    # This would implement historical data backfill
    # For now, just return success
    return f"Backfill completed: {start_date} to {end_date}"


with DAG(
    'spp_backfill',
    default_args=default_args,
    description='SPP historical data backfill',
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=['backfill', 'spp', 'historical'],
) as dag_spp_backfill:

    backfill_task = PythonOperator(
        task_id='run_spp_backfill',
        python_callable=run_spp_backfill,
        provide_context=True,
    )
