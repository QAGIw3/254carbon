"""
AESO Ingestion DAGs

Overview
--------
Schedules ingestion for:
- Pool Price (hourly)
- Alberta Internal Load (every 5 minutes)
- Intertie flows (every 5 minutes)
- Generation mix (every 5 minutes)

Design
------
- Each DAG wraps a PythonOperator that instantiates ``AESOConnector`` with the
  appropriate ``data_type`` and configuration, then runs the connector.
- Basic quality checks assert minimum records processed and staleness bounds in
  ClickHouse by inspecting the latest ``event_time``.

Auth & Runtime Configuration
----------------------------
- Live AESO API calls can be enabled per connector (POOL/AIL) via flags
  ``use_live_pool`` and ``use_live_ail``.
- Secrets are sourced from environment or Airflow Variables:
  ``AESO_BEARER_TOKEN`` or ``AESO_API_KEY``.

Data Lineage
------------
AESO API → Connector → Kafka (e.g., ``power.ticks.v1``/``power.load.v1``) →
ClickHouse downstream tables for analytics.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from aeso_connector import AESOConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_aeso_pool_ingestion(**context):
    """Execute AESO Pool Price ingestion (live if enabled)."""
    config = {
        'source_id': 'aeso_pool_live',
        'data_type': 'POOL',
        'use_live_pool': True,  # Toggle live API
        'api_base': os.getenv('AESO_API_BASE', 'https://api.aeso.ca/report/v1'),
        # Pass auth explicitly; connector also reads env
        'bearer_token': os.getenv('AESO_BEARER_TOKEN', Variable.get('AESO_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = AESOConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_aeso_ail_ingestion(**context):
    """Execute AESO AIL ingestion (live if enabled)."""
    config = {
        'source_id': 'aeso_ail_live',
        'data_type': 'AIL',
        'use_live_ail': True,  # Toggle live API
        'api_base': os.getenv('AESO_API_BASE', 'https://api.aeso.ca/report/v1'),
        'bearer_token': os.getenv('AESO_BEARER_TOKEN', Variable.get('AESO_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.load.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = AESOConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def check_aeso_pool_quality(**context):
    """
    Validate data quality after AESO Pool Price ingestion.

    Criteria
    - Non-zero events processed reported via XCom
    - Latest energy record for AESO in ClickHouse not older than 2 hours
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_aeso_pool', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No pool price events processed: {events_processed}")

    # Optional staleness check via DB (implementation dependent on schema access)
    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    result = pg_hook.get_first(
        """
        SELECT MAX(event_time) as latest
        FROM ch.market_price_ticks
        WHERE source LIKE 'aeso%'
          AND product = 'energy'
        """
    )
    if result:
        latest_time = result[0]
        if latest_time and (datetime.utcnow() - latest_time).total_seconds() > 7200:
            raise ValueError(f"AESO Pool Price staleness detected: latest={latest_time}")

    return True


def check_aeso_ail_quality(**context):
    """
    Validate data quality after AESO AIL ingestion.

    Criteria
    - Non-zero events processed
    - Latest load record not older than 15 minutes
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_aeso_ail', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No AIL events processed: {events_processed}")

    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    result = pg_hook.get_first(
        """
        SELECT MAX(event_time) as latest
        FROM ch.market_price_ticks
        WHERE source LIKE 'aeso%'
          AND product = 'load'
        """
    )
    if result:
        latest_time = result[0]
        if latest_time and (datetime.utcnow() - latest_time).total_seconds() > 900:
            raise ValueError(f"AESO AIL staleness detected: latest={latest_time}")

    return True


def run_aeso_intertie_ingestion(**context):
    """
    Execute AESO Intertie ingestion (mock by default).

    If live endpoints are supplied in the connector in the future, this
    task can be toggled similarly to Pool/AIL.
    """
    config = {
        'source_id': 'aeso_intertie',
        'data_type': 'INTERTIE',
        'api_base': os.getenv('AESO_API_BASE', 'https://api.aeso.ca/report/v1'),
        'kafka_topic': 'power.transmission.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = AESOConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_aeso_generation_ingestion(**context):
    """
    Execute AESO Generation mix ingestion (mock by default).

    Mirrors the approach for Intertie; focuses on data pipeline plumbing
    and downstream consistency in the canonical schema.
    """
    config = {
        'source_id': 'aeso_generation',
        'data_type': 'GENERATION',
        'api_base': os.getenv('AESO_API_BASE', 'https://api.aeso.ca/report/v1'),
        'kafka_topic': 'power.generation.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = AESOConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def check_aeso_intertie_quality(**context):
    """
    Validate data quality after AESO Intertie ingestion.

    Criteria
    - Non-zero events processed
    - Latest transmission record not older than 15 minutes
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_aeso_intertie', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No intertie events processed: {events_processed}")

    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    result = pg_hook.get_first(
        """
        SELECT MAX(event_time) as latest
        FROM ch.market_price_ticks
        WHERE source LIKE 'aeso%'
          AND product = 'transmission'
        """
    )
    if result:
        latest_time = result[0]
        if latest_time and (datetime.utcnow() - latest_time).total_seconds() > 900:
            raise ValueError(f"AESO Intertie staleness detected: latest={latest_time}")

    return True


def check_aeso_generation_quality(**context):
    """
    Validate data quality after AESO Generation ingestion.

    Criteria
    - Non-zero events processed
    - Latest generation record not older than 30 minutes
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_aeso_generation', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No generation events processed: {events_processed}")

    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    result = pg_hook.get_first(
        """
        SELECT MAX(event_time) as latest
        FROM ch.market_price_ticks
        WHERE source LIKE 'aeso%'
          AND product = 'generation'
        """
    )
    if result:
        latest_time = result[0]
        if latest_time and (datetime.utcnow() - latest_time).total_seconds() > 1800:
            raise ValueError(f"AESO Generation staleness detected: latest={latest_time}")

    return True


# Pool Price ingestion (hourly)
with DAG(
    'aeso_pool_ingestion',
    default_args=default_args,
    description='AESO Pool Price ingestion',
    schedule_interval='0 * * * *',  # hourly
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'aeso', 'pool'],
) as dag_pool:

    ingest_task = PythonOperator(
        task_id='ingest_aeso_pool',
        python_callable=run_aeso_pool_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_aeso_pool_quality',
        python_callable=check_aeso_pool_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# AIL ingestion (every 5 minutes)
with DAG(
    'aeso_ail_ingestion',
    default_args=default_args,
    description='AESO Alberta Internal Load ingestion',
    schedule_interval='*/5 * * * *',  # every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'aeso', 'ail', 'load'],
) as dag_ail:

    ingest_task = PythonOperator(
        task_id='ingest_aeso_ail',
        python_callable=run_aeso_ail_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_aeso_ail_quality',
        python_callable=check_aeso_ail_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# Intertie ingestion (every 5 minutes)
with DAG(
    'aeso_intertie_ingestion',
    default_args=default_args,
    description='AESO Intertie flows ingestion',
    schedule_interval='*/5 * * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'aeso', 'intertie', 'transmission'],
) as dag_intertie:

    ingest_task = PythonOperator(
        task_id='ingest_aeso_intertie',
        python_callable=run_aeso_intertie_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_aeso_intertie_quality',
        python_callable=check_aeso_intertie_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# Generation mix ingestion (every 5 minutes)
with DAG(
    'aeso_generation_ingestion',
    default_args=default_args,
    description='AESO Generation mix ingestion',
    schedule_interval='*/5 * * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'aeso', 'generation'],
) as dag_generation:

    ingest_task = PythonOperator(
        task_id='ingest_aeso_generation',
        python_callable=run_aeso_generation_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_aeso_generation_quality',
        python_callable=check_aeso_generation_quality,
        provide_context=True,
    )

    ingest_task >> quality_check
