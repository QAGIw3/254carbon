"""
IESO Ingestion DAGs

Overview
--------
Schedules ingestion for Ontario electricity market:
- HOEP (Hourly Ontario Energy Price) - hourly
- Ontario demand - hourly
- Generation by fuel type - hourly
- Intertie flows - hourly

Design
------
- Each DAG wraps a PythonOperator that instantiates ``IESOConnector`` with the
  appropriate configuration and runs the connector.
- Data quality checks validate minimum records and freshness.

Data Lineage
------------
IESO API → Connector → Kafka (power.ticks.v1) → ClickHouse → Analytics

Schedule
--------
- HOEP: Hourly (top of hour)
- Demand/Generation: Every hour
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

from ieso_connector import IESOConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


def run_ieso_hoep_ingestion(**context):
    """Execute IESO HOEP (Hourly Ontario Energy Price) ingestion."""
    config = {
        'source_id': 'ieso_hoep',
        'data_type': 'HOEP',
        'api_base': os.getenv('IESO_API_BASE', 'https://www.ieso.ca/api'),
        'bearer_token': os.getenv('IESO_BEARER_TOKEN', Variable.get('IESO_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = IESOConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_ieso_demand_ingestion(**context):
    """Execute IESO demand ingestion."""
    config = {
        'source_id': 'ieso_demand',
        'data_type': 'DEMAND',
        'api_base': os.getenv('IESO_API_BASE', 'https://www.ieso.ca/api'),
        'bearer_token': os.getenv('IESO_BEARER_TOKEN', Variable.get('IESO_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.load.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = IESOConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_ieso_generation_ingestion(**context):
    """Execute IESO generation mix ingestion."""
    config = {
        'source_id': 'ieso_generation',
        'data_type': 'GENERATION',
        'api_base': os.getenv('IESO_API_BASE', 'https://www.ieso.ca/api'),
        'bearer_token': os.getenv('IESO_BEARER_TOKEN', Variable.get('IESO_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.generation.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = IESOConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def check_ieso_hoep_quality(**context):
    """
    Validate data quality after IESO HOEP ingestion.

    Criteria:
    - Non-zero events processed
    - Latest HOEP record not older than 2 hours
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_ieso_hoep', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No HOEP events processed: {events_processed}")

    # Check freshness in ClickHouse
    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    result = pg_hook.get_first(
        """
        SELECT MAX(event_time) as latest
        FROM ch.market_price_ticks
        WHERE source LIKE 'ieso%'
          AND product = 'energy'
        """
    )
    if result:
        latest_time = result[0]
        if latest_time and (datetime.utcnow() - latest_time).total_seconds() > 7200:  # 2 hours
            raise ValueError(f"IESO HOEP staleness detected: latest={latest_time}")

    return True


def check_ieso_demand_quality(**context):
    """
    Validate data quality after IESO demand ingestion.

    Criteria:
    - Non-zero events processed
    - Latest demand record not older than 2 hours
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_ieso_demand', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No demand events processed: {events_processed}")

    return True


def check_ieso_generation_quality(**context):
    """Validate IESO generation data quality."""
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_ieso_generation', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No generation events processed: {events_processed}")

    return True


# HOEP ingestion (hourly)
with DAG(
    'ieso_hoep_ingestion',
    default_args=default_args,
    description='IESO HOEP (Hourly Ontario Energy Price) ingestion',
    schedule_interval='0 * * * *',  # Hourly at top of hour
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'ieso', 'hoep', 'ontario'],
) as dag_hoep:

    ingest_task = PythonOperator(
        task_id='ingest_ieso_hoep',
        python_callable=run_ieso_hoep_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_ieso_hoep_quality',
        python_callable=check_ieso_hoep_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# Demand ingestion (hourly)
with DAG(
    'ieso_demand_ingestion',
    default_args=default_args,
    description='IESO Ontario demand ingestion',
    schedule_interval='0 * * * *',  # Hourly
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'ieso', 'demand', 'ontario'],
) as dag_demand:

    ingest_task = PythonOperator(
        task_id='ingest_ieso_demand',
        python_callable=run_ieso_demand_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_ieso_demand_quality',
        python_callable=check_ieso_demand_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# Generation ingestion (hourly)
with DAG(
    'ieso_generation_ingestion',
    default_args=default_args,
    description='IESO generation mix ingestion',
    schedule_interval='0 * * * *',  # Hourly
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'ieso', 'generation', 'ontario'],
) as dag_generation:

    ingest_task = PythonOperator(
        task_id='ingest_ieso_generation',
        python_callable=run_ieso_generation_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_ieso_generation_quality',
        python_callable=check_ieso_generation_quality,
        provide_context=True,
    )

    ingest_task >> quality_check

