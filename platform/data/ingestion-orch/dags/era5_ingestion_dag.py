"""
ERA5 (ECMWF Reanalysis) Ingestion DAG

Runs ERA5 live ingestion via ERA5Connector to publish hourly weather
variables to Kafka fundamentals. Includes optional backfill in day-sized
chunks between configured dates to keep NetCDF downloads manageable.
"""
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))
from external.weather.era5_connector import ERA5Connector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def _parse_list(val: str | None) -> list[str]:
    if not val:
        return []
    return [x.strip() for x in val.split(',') if x.strip()]


def _parse_area(val: str | None) -> list[float] | None:
    if not val:
        return None
    try:
        parts = [float(x.strip()) for x in val.split(',')]
        if len(parts) == 4:
            return parts
    except Exception:
        return None
    return None


def run_era5_live(**context):
    dataset = os.getenv('ERA5_DATASET', Variable.get('ERA5_DATASET', default_var='reanalysis-era5-single-levels'))
    variables = _parse_list(os.getenv('ERA5_VARIABLES', Variable.get('ERA5_VARIABLES', default_var='2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,total_precipitation')))
    area = _parse_area(os.getenv('ERA5_AREA', Variable.get('ERA5_AREA', default_var=None)))
    hours = _parse_list(os.getenv('ERA5_HOURS', Variable.get('ERA5_HOURS', default_var='')))
    lat = os.getenv('ERA5_LATITUDE', Variable.get('ERA5_LATITUDE', default_var=None))
    lon = os.getenv('ERA5_LONGITUDE', Variable.get('ERA5_LONGITUDE', default_var=None))

    cfg = {
        'source_id': 'era5_live',
        'live': True,
        'dataset': dataset,
        'variables': variables or None,
        'area': area,
        'hours': hours or None,
        'latitude': float(lat) if lat else None,
        'longitude': float(lon) if lon else None,
        'cds_url': os.getenv('CDS_URL', Variable.get('CDS_URL', default_var=None)),
        'cds_key': os.getenv('CDS_KEY', Variable.get('CDS_KEY', default_var=None)),
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }

    connector = ERA5Connector(cfg)
    return connector.run()


def run_era5_backfill(**context):
    start = os.getenv('ERA5_BACKFILL_START', Variable.get('ERA5_BACKFILL_START', default_var=None))
    end = os.getenv('ERA5_BACKFILL_END', Variable.get('ERA5_BACKFILL_END', default_var=None))
    if not start or not end:
        return 0

    dataset = os.getenv('ERA5_DATASET', Variable.get('ERA5_DATASET', default_var='reanalysis-era5-single-levels'))
    variables = _parse_list(os.getenv('ERA5_VARIABLES', Variable.get('ERA5_VARIABLES', default_var='2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,total_precipitation')))
    area = _parse_area(os.getenv('ERA5_AREA', Variable.get('ERA5_AREA', default_var=None)))
    hours = _parse_list(os.getenv('ERA5_HOURS', Variable.get('ERA5_HOURS', default_var='')))
    lat = os.getenv('ERA5_LATITUDE', Variable.get('ERA5_LATITUDE', default_var=None))
    lon = os.getenv('ERA5_LONGITUDE', Variable.get('ERA5_LONGITUDE', default_var=None))
    cds_url = os.getenv('CDS_URL', Variable.get('CDS_URL', default_var=None))
    cds_key = os.getenv('CDS_KEY', Variable.get('CDS_KEY', default_var=None))

    chunk_days = int(os.getenv('ERA5_BACKFILL_CHUNK_DAYS', Variable.get('ERA5_BACKFILL_CHUNK_DAYS', default_var='1')))
    total = 0
    cur = datetime.fromisoformat(start).date()
    end_dt = datetime.fromisoformat(end).date()
    from datetime import timedelta as _td

    while cur <= end_dt:
        chunk_end = min(cur + _td(days=chunk_days - 1), end_dt)
        cfg = {
            'source_id': f'era5_backfill_{cur.isoformat()}_{chunk_end.isoformat()}',
            'live': True,
            'dataset': dataset,
            'variables': variables or None,
            'area': area,
            'hours': hours or None,
            'start_date': cur.isoformat(),
            'end_date': chunk_end.isoformat(),
            'latitude': float(lat) if lat else None,
            'longitude': float(lon) if lon else None,
            'cds_url': cds_url,
            'cds_key': cds_key,
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
        }
        connector = ERA5Connector(cfg)
        total += connector.run()
        cur = chunk_end + _td(days=1)

    return total


with DAG(
    dag_id='era5_ingestion',
    default_args=default_args,
    description='Ingest ERA5 (ECMWF) hourly reanalysis variables to Kafka fundamentals',
    schedule_interval=os.getenv('ERA5_SCHEDULE', '0 3 * * *'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    backfill = PythonOperator(
        task_id='backfill_era5',
        python_callable=run_era5_backfill,
        provide_context=True,
    )
    ingest = PythonOperator(
        task_id='ingest_era5',
        python_callable=run_era5_live,
        provide_context=True,
    )

    backfill >> ingest

