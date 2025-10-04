"""
CAISO Real-Time and Day-Ahead LMP Ingestion DAGs

Schedules
- RTM: every 5 minutes
- DAM: hourly

Design
- Uses CAISOConnector which fetches OASIS SingleZip CSV-in-ZIP payloads,
  parses to canonical schema, and enforces hub-only pilot restrictions.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from caiso_connector import CAISOConnector

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_caiso_rtm_ingestion(**context):
    """Execute CAISO real-time market LMP ingestion with entitlement restrictions."""
    config = {
        'source_id': 'caiso_rtm_lmp',
        'market_type': 'RTM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
        'hub_only': True,  # Pilot restriction: hub data only
        'entitlements_enabled': True,
        'timeout_seconds': 30,
        'max_retries': 3,
        'retry_backoff_base': 1.0,
        'dev_mode': False,
    }
    
    connector = CAISOConnector(config)
    events_processed = connector.run()
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    
    return events_processed


def run_caiso_dam_ingestion(**context):
    """Execute CAISO day-ahead market LMP ingestion with entitlement restrictions."""
    config = {
        'source_id': 'caiso_dam_lmp',
        'market_type': 'DAM',
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
        'hub_only': True,  # Pilot restriction: hub data only
        'entitlements_enabled': True,
        'timeout_seconds': 30,
        'max_retries': 3,
        'retry_backoff_base': 1.0,
        'dev_mode': False,
    }
    
    connector = CAISOConnector(config)
    events_processed = connector.run()
    
    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    
    return events_processed


def check_caiso_data_quality(**context):
    """Validate data quality of ingested CAISO data."""
    ti = context['task_instance']
    # Allow both RTM and DAM checks by probing known task ids
    events_processed = (
        ti.xcom_pull(task_ids='ingest_caiso_rtm', key='events_processed')
        or ti.xcom_pull(task_ids='ingest_caiso_dam', key='events_processed')
    )
    
    if events_processed is None or events_processed < 3:
        raise ValueError(f"Too few events processed: {events_processed} (expected at least 3 hubs)")
    
    # Additional quality checks
    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    
    # Check for data freshness
    result = pg_hook.get_first("""
        SELECT MAX(event_time) as latest
        FROM ch.market_price_ticks
        WHERE source LIKE 'caiso%'
    """)
    
    if result:
        latest_time = result[0]
        if latest_time and (datetime.utcnow() - latest_time).total_seconds() > 900:
            raise ValueError(f"Data staleness detected: latest={latest_time}")
    
    return True


def verify_entitlement_restrictions(**context):
    """Verify that only hub data is being ingested (pilot restriction)."""
    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    
    # Check that only trading hubs are in the data
    allowed_hubs = [
        'CAISO.TH_SP15_GEN-APND',
        'CAISO.TH_NP15_GEN-APND',
        'CAISO.TH_ZP26_GEN-APND',
    ]
    
    result = pg_hook.get_records("""
        SELECT DISTINCT instrument_id
        FROM ch.market_price_ticks
        WHERE source LIKE 'caiso%'
          AND event_time > NOW() - INTERVAL '1 hour'
    """)
    
    if result:
        instrument_ids = [row[0] for row in result]
        unauthorized = [iid for iid in instrument_ids if iid not in allowed_hubs]
        
        if unauthorized:
            raise ValueError(f"Unauthorized instruments detected: {unauthorized}")
    
    return True


# Real-time market ingestion DAG (every 5 minutes)
with DAG(
    'caiso_rtm_ingestion',
    default_args=default_args,
    description='CAISO real-time market LMP data ingestion (hub-only for pilot)',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'caiso', 'realtime', 'pilot'],
) as dag_rtm:
    
    ingest_task = PythonOperator(
        task_id='ingest_caiso_rtm',
        python_callable=run_caiso_rtm_ingestion,
        provide_context=True,
    )
    
    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_caiso_data_quality,
        provide_context=True,
    )
    
    entitlement_check = PythonOperator(
        task_id='verify_entitlements',
        python_callable=verify_entitlement_restrictions,
        provide_context=True,
    )
    
    ingest_task >> quality_check >> entitlement_check


# Day-ahead market ingestion DAG (hourly)
with DAG(
    'caiso_dam_ingestion',
    default_args=default_args,
    description='CAISO day-ahead market LMP data ingestion (hub-only for pilot)',
    schedule_interval='0 * * * *',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'caiso', 'day-ahead', 'pilot'],
) as dag_dam:

    ingest_task = PythonOperator(
        task_id='ingest_caiso_dam',
        python_callable=run_caiso_dam_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_caiso_data_quality,
        provide_context=True,
    )

    entitlement_check = PythonOperator(
        task_id='verify_entitlements',
        python_callable=verify_entitlement_restrictions,
        provide_context=True,
    )

    ingest_task >> quality_check >> entitlement_check
