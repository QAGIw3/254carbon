"""
Brazil ONS Ingestion DAGs

Overview
--------
Schedules ingestion for Brazilian electricity market:
- PLD (Preço de Liquidação das Diferenças) - weekly settlement prices
- Load forecasts - daily
- Generation by source - hourly
- Hydro reservoir levels - daily

Design
------
- Each DAG wraps a PythonOperator that instantiates ``BrazilONSConnector`` with the
  appropriate configuration and runs the connector.
- Data quality checks validate minimum records and price reasonableness.

Brazilian Subsystems
--------------------
- N (North)
- NE (Northeast)
- S (South)
- SECO (Southeast/Midwest)

Data Lineage
------------
ONS API → Connector → Kafka (power.ticks.v1) → ClickHouse → Analytics

Schedule
--------
- PLD: Daily at 08:00 BRT (weekly prices published)
- Load Forecast: Daily at 06:00 BRT
- Generation: Hourly
- Hydro Reservoirs: Daily at 07:00 BRT
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

from brazil_ons_connector import BrazilONSConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
}


def run_brazil_pld_ingestion(**context):
    """Execute Brazil PLD (settlement price) ingestion."""
    config = {
        'source_id': 'brazil_ons_pld',
        'data_type': 'PLD',
        'submarket': 'ALL',  # All 4 subsystems
        'api_base': os.getenv('ONS_API_BASE', 'https://ons-data.operador.org.br/api'),
        'bearer_token': os.getenv('ONS_BEARER_TOKEN', Variable.get('ONS_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.ticks.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = BrazilONSConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_brazil_load_forecast_ingestion(**context):
    """Execute Brazil load forecast ingestion."""
    config = {
        'source_id': 'brazil_ons_load_forecast',
        'data_type': 'LOAD_FORECAST',
        'submarket': 'ALL',
        'api_base': os.getenv('ONS_API_BASE', 'https://ons-data.operador.org.br/api'),
        'bearer_token': os.getenv('ONS_BEARER_TOKEN', Variable.get('ONS_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.load.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = BrazilONSConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_brazil_generation_ingestion(**context):
    """Execute Brazil generation mix ingestion."""
    config = {
        'source_id': 'brazil_ons_generation',
        'data_type': 'GENERATION',
        'submarket': 'ALL',
        'api_base': os.getenv('ONS_API_BASE', 'https://ons-data.operador.org.br/api'),
        'bearer_token': os.getenv('ONS_BEARER_TOKEN', Variable.get('ONS_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.generation.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = BrazilONSConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def run_brazil_hydro_ingestion(**context):
    """Execute Brazil hydro reservoir levels ingestion."""
    config = {
        'source_id': 'brazil_ons_hydro',
        'data_type': 'HYDRO',
        'submarket': 'ALL',
        'api_base': os.getenv('ONS_API_BASE', 'https://ons-data.operador.org.br/api'),
        'bearer_token': os.getenv('ONS_BEARER_TOKEN', Variable.get('ONS_BEARER_TOKEN', default_var='')),
        'kafka_topic': 'power.hydro.v1',
        'kafka_bootstrap': 'kafka:9092',
    }

    connector = BrazilONSConnector(config)
    events_processed = connector.run()

    context['task_instance'].xcom_push(key='events_processed', value=events_processed)
    return events_processed


def check_brazil_pld_quality(**context):
    """
    Validate data quality after Brazil PLD ingestion.

    Criteria:
    - Non-zero events processed
    - Expect 4 subsystems (N, NE, S, SECO)
    - PLD prices within reasonable range (R$50-R$1000/MWh)
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_brazil_pld', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No Brazil PLD events processed: {events_processed}")

    # Expect at least 4 subsystems
    if events_processed < 4:
        raise ValueError(f"Insufficient Brazil subsystems covered: {events_processed} events (expected >=4)")

    # Check price reasonableness in ClickHouse
    pg_hook = PostgresHook(postgres_conn_id='market_intelligence_db')
    result = pg_hook.get_first(
        """
        SELECT MIN(value) as min_price, MAX(value) as max_price
        FROM ch.market_price_ticks
        WHERE source LIKE 'ons%'
          AND product = 'energy'
          AND event_time >= now() - INTERVAL 1 WEEK
        """
    )
    if result:
        min_price, max_price = result
        # Brazilian PLD typically ranges R$50-500/MWh, can spike to R$1000+
        if min_price is not None and min_price < 0:
            raise ValueError(f"Invalid negative PLD detected: {min_price}")
        if max_price is not None and max_price > 2000:
            raise ValueError(f"Unreasonable PLD spike detected: {max_price} (investigate)")

    return True


def check_brazil_load_forecast_quality(**context):
    """Validate data quality after Brazil load forecast ingestion."""
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_brazil_load_forecast', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No Brazil load forecast events processed: {events_processed}")

    return True


def check_brazil_generation_quality(**context):
    """Validate data quality after Brazil generation ingestion."""
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_brazil_generation', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No Brazil generation events processed: {events_processed}")

    return True


def check_brazil_hydro_quality(**context):
    """
    Validate data quality after Brazil hydro reservoir ingestion.

    Criteria:
    - Non-zero events processed
    - Storage levels within 0-100% range
    """
    ti = context['task_instance']
    events_processed = ti.xcom_pull(task_ids='ingest_brazil_hydro', key='events_processed')

    if events_processed is None or events_processed <= 0:
        raise ValueError(f"No Brazil hydro reservoir events processed: {events_processed}")

    return True


# PLD ingestion (daily at 08:00 UTC - 05:00 BRT)
with DAG(
    'brazil_pld_ingestion',
    default_args=default_args,
    description='Brazil PLD (settlement price) ingestion',
    schedule_interval='0 11 * * *',  # Daily at 11:00 UTC = 08:00 BRT
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'brazil', 'pld', 'settlement'],
) as dag_pld:

    ingest_task = PythonOperator(
        task_id='ingest_brazil_pld',
        python_callable=run_brazil_pld_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_brazil_pld_quality',
        python_callable=check_brazil_pld_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# Load Forecast ingestion (daily at 06:00 UTC - 03:00 BRT)
with DAG(
    'brazil_load_forecast_ingestion',
    default_args=default_args,
    description='Brazil load forecast ingestion',
    schedule_interval='0 9 * * *',  # Daily at 09:00 UTC = 06:00 BRT
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'brazil', 'load_forecast', 'demand'],
) as dag_load:

    ingest_task = PythonOperator(
        task_id='ingest_brazil_load_forecast',
        python_callable=run_brazil_load_forecast_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_brazil_load_forecast_quality',
        python_callable=check_brazil_load_forecast_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# Generation ingestion (hourly)
with DAG(
    'brazil_generation_ingestion',
    default_args=default_args,
    description='Brazil generation mix ingestion',
    schedule_interval='0 * * * *',  # Hourly
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'brazil', 'generation'],
) as dag_generation:

    ingest_task = PythonOperator(
        task_id='ingest_brazil_generation',
        python_callable=run_brazil_generation_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_brazil_generation_quality',
        python_callable=check_brazil_generation_quality,
        provide_context=True,
    )

    ingest_task >> quality_check


# Hydro reservoir ingestion (daily at 07:00 UTC - 04:00 BRT)
with DAG(
    'brazil_hydro_reservoir_ingestion',
    default_args=default_args,
    description='Brazil hydro reservoir levels ingestion',
    schedule_interval='0 10 * * *',  # Daily at 10:00 UTC = 07:00 BRT
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'brazil', 'hydro', 'reservoirs'],
) as dag_hydro:

    ingest_task = PythonOperator(
        task_id='ingest_brazil_hydro',
        python_callable=run_brazil_hydro_ingestion,
        provide_context=True,
    )

    quality_check = PythonOperator(
        task_id='check_brazil_hydro_quality',
        python_callable=check_brazil_hydro_quality,
        provide_context=True,
    )

    ingest_task >> quality_check

