"""
NOAA CDO Ingestion DAG

Runs NOAA CDO live ingestion using NOAACDOConnector with token/header.
Schedule: daily at 02:00 UTC by default.
"""
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))
from external.weather.noaa_cdo_connector import NOAACDOConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def run_noaa_cdo_live(**context):
    cfg = {
        'source_id': 'noaa_cdo_live',
        'live': True,
        'api_base': os.getenv('NOAA_CDO_API_BASE', 'https://www.ncdc.noaa.gov/cdo-web/api/v2'),
        'token': os.getenv('NOAA_CDO_TOKEN', Variable.get('NOAA_CDO_TOKEN', default_var='')),
        # Example: CA statewide, daily TAVG/PRCP over last 7 days
        'datasetid': os.getenv('NOAA_CDO_DATASET', Variable.get('NOAA_CDO_DATASET', default_var='GHCND')),
        'stationid': os.getenv('NOAA_CDO_STATIONID', Variable.get('NOAA_CDO_STATIONID', default_var=None)),
        'locationid': os.getenv('NOAA_CDO_LOCATIONID', Variable.get('NOAA_CDO_LOCATIONID', default_var='FIPS:06')),
        'datatypeid': os.getenv('NOAA_CDO_DATATYPES', 'TAVG,PRCP').split(','),
        'startdate': os.getenv('NOAA_CDO_STARTDATE'),
        'enddate': os.getenv('NOAA_CDO_ENDDATE'),
        'limit': int(os.getenv('NOAA_CDO_LIMIT', '1000')),
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }

    connector = NOAACDOConnector(cfg)
    return connector.run()


def run_noaa_cdo_backfill(**context):
    """Backfill NOAA CDO daily data in monthly chunks between two dates.

    Env inputs:
    - NOAA_CDO_BACKFILL_START (YYYY-MM-DD)
    - NOAA_CDO_BACKFILL_END (YYYY-MM-DD)
    """
    from datetime import datetime, timedelta
    import dateutil.parser as dp

    start = os.getenv('NOAA_CDO_BACKFILL_START', Variable.get('NOAA_CDO_BACKFILL_START', default_var=None))
    end = os.getenv('NOAA_CDO_BACKFILL_END', Variable.get('NOAA_CDO_BACKFILL_END', default_var=None))
    if not start or not end:
        return 0

    start_dt = dp.parse(start).date()
    end_dt = dp.parse(end).date()

    total = 0
    cur = start_dt
    while cur <= end_dt:
        # chunk: 31 days window
        chunk_end = min(cur + timedelta(days=30), end_dt)
        cfg = {
            'source_id': 'noaa_cdo_backfill',
            'live': True,
            'api_base': os.getenv('NOAA_CDO_API_BASE', 'https://www.ncdc.noaa.gov/cdo-web/api/v2'),
            'token': os.getenv('NOAA_CDO_TOKEN', Variable.get('NOAA_CDO_TOKEN', default_var='')),
            'datasetid': os.getenv('NOAA_CDO_DATASET', Variable.get('NOAA_CDO_DATASET', default_var='GHCND')),
            'stationid': os.getenv('NOAA_CDO_STATIONID', Variable.get('NOAA_CDO_STATIONID', default_var=None)),
            'locationid': os.getenv('NOAA_CDO_LOCATIONID', Variable.get('NOAA_CDO_LOCATIONID', default_var='FIPS:06')),
            'datatypeid': os.getenv('NOAA_CDO_DATATYPES', 'TAVG,PRCP').split(','),
            'startdate': cur.isoformat(),
            'enddate': chunk_end.isoformat(),
            'limit': int(os.getenv('NOAA_CDO_LIMIT', '1000')),
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
        }
        connector = NOAACDOConnector(cfg)
        total += connector.run()
        cur = chunk_end + timedelta(days=1)
    return total


with DAG(
    dag_id='noaa_cdo_ingestion',
    default_args=default_args,
    description='Ingest NOAA CDO data to Kafka fundamentals',
    schedule_interval=os.getenv('NOAA_CDO_SCHEDULE', '0 2 * * *'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    ingest_task = PythonOperator(
        task_id='ingest_noaa_cdo',
        python_callable=run_noaa_cdo_live,
        provide_context=True,
    )
    backfill_task = PythonOperator(
        task_id='backfill_noaa_cdo',
        python_callable=run_noaa_cdo_backfill,
        provide_context=True,
    )

    backfill_task >> ingest_task
