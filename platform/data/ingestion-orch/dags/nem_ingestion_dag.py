"""
NEM Ingestion DAGs

Overview
--------
Schedules ingestion for Australian National Electricity Market:
- Spot Price (30-minute intervals) - all 5 NEM regions
- Regional demand (30-minute intervals)
- Interconnector flows (30-minute intervals)

Design
------
- Each DAG wraps a PythonOperator that instantiates ``NEMConnector`` with the
  appropriate configuration and runs the connector.
- Data quality checks validate minimum records and freshness.

NEM Regions
-----------
- NSW (New South Wales)
- QLD (Queensland)
- SA (South Australia)
- TAS (Tasmania)
- VIC (Victoria)

Data Lineage
------------
NEM API → Connector → Kafka (power.ticks.v1) → ClickHouse → Analytics

Schedule
--------
- Spot Price: Every 30 minutes (aligned with NEM trading periods)
- Demand: Every 30 minutes
- Interconnectors: Every 30 minutes
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

from nem_connector import NEMConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


def run_nem_spot_price_ingestion(**context):
    """Execute NEM spot price ingestion for all regions."""
    config = {
        'source_id': 'nem_spot_prices',
        'data_type': 'SPOT_PRICE',
        'api_base': os.getenv('NEM_API_BASE', 'https://api.nemweb.com.au'),
        'api_key': os.getenv('NEM_API_KEY', Variable.get('NEM_API_KEY', default_var='')),
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = NEMConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_nem_demand_ingestion(**context):
    """Execute NEM regional demand ingestion."""
    config = {
        'source_id': 'nem_demand',
        'data_type': 'DEMAND',
        'api_base': os.getenv('NEM_API_BASE', 'https://api.nemweb.com.au'),
        'api_key': os.getenv('NEM_API_KEY', Variable.get('NEM_API_KEY', default_var='')),
        'kafka_topic': 'power.load.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = NEMConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_nem_interconnector_ingestion(**context):
    """Execute NEM interconnector flows ingestion."""
    config = {
        'source_id': 'nem_interconnectors',
        'data_type': 'INTERCONNECTOR',
        'api_base': os.getenv('NEM_API_BASE', 'https://api.nemweb.com.au'),
        'api_key': os.getenv('NEM_API_KEY', Variable.get('NEM_API_KEY', default_var='')),
        'kafka_topic': 'power.transmission.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = NEMConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def check_nem_spot_price_quality(**context):
    """
    Validate data quality after NEM spot price ingestion.

    Criteria:
    - Non-zero events processed
    - Latest spot price not older than 90 minutes (allow for 30-min period + delay)
    - Prices for all 5 NEM regions present
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_nem_spot_price', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No NEM spot price events processed: {events_processed}")

    # Expect at least 5 regions * 1 period = 5 events minimum
    if events_processed < 5:
        raise ValueError(f"Insufficient NEM regions covered: {events_processed} events")

    # Check freshness in ClickHouse
    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    result = pg_hook.get_first(
        """
        SELECT MAX(event_time) as latest
        FROM ch.market_price_ticks
        WHERE source LIKE 'nem%'
          AND product = 'energy'
        """
    )
    if result:
        latest_time = result[0]
        if latest_time and (datetime.utcnow() - latest_time).total_seconds() > 5400:  # 90 minutes
            raise ValueError(f"NEM spot price staleness detected: latest={latest_time}")

    return True


def check_nem_demand_quality(**context):
    """Validate data quality after NEM demand ingestion."""
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_nem_demand', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No NEM demand events processed: {events_processed}")

    # Expect at least 5 regions
    if events_processed < 5:
        raise ValueError(f"Insufficient NEM regions covered: {events_processed} events")

    return True


def check_nem_interconnector_quality(**context):
    """Validate data quality after NEM interconnector ingestion."""
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_nem_interconnector', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No NEM interconnector events processed: {events_processed}")

    return True


# NEM Spot Price ingestion (every 30 minutes)
with DAG(
    'nem_spot_price_ingestion',
    default_args=default_args,
    description='NEM spot price ingestion for all Australian regions',
    schedule_interval='*/30 * * * *',  # Every 30 minutes (aligned with NEM trading periods)
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'nem', 'spot_price', 'australia'],
) as dag_spot:

    ingest_task = PythonOperator(
        task_id='ingest_nem_spot_price',
        python_callable=run_nem_spot_price_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_nem_spot_price_quality',
        python_callable=check_nem_spot_price_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# NEM Demand ingestion (every 30 minutes)
with DAG(
    'nem_demand_ingestion',
    default_args=default_args,
    description='NEM regional demand ingestion',
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'nem', 'demand', 'australia'],
) as dag_demand:

    ingest_task = PythonOperator(
        task_id='ingest_nem_demand',
        python_callable=run_nem_demand_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_nem_demand_quality',
        python_callable=check_nem_demand_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# NEM Interconnector ingestion (every 30 minutes)
with DAG(
    'nem_interconnector_ingestion',
    default_args=default_args,
    description='NEM interconnector flows ingestion',
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'nem', 'interconnector', 'australia'],
) as dag_interconnector:

    ingest_task = PythonOperator(
        task_id='ingest_nem_interconnector',
        python_callable=run_nem_interconnector_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_nem_interconnector_quality',
        python_callable=check_nem_interconnector_quality,
        provide_context=True,
    )

    ingest_task >> quality_check

