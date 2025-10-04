"""
ENTSO-E Transparency Ingestion DAG

Ingests European grid fundamentals (load, generation, flows, day-ahead price)
using ENTSOETransparencyConnector. Supports optional backfill windows.
"""
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))
from external.infrastructure.entsoe_connector import ENTSOETransparencyConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def _get_json_map(var_name: str) -> dict:
    val = os.getenv(var_name, Variable.get(var_name, default_var=''))
    if not val:
        return {}
    try:
        return json.loads(val)
    except Exception:
        return {}


def run_entsoe_live(**context):
    api_base = os.getenv('ENTSOE_API_BASE', Variable.get('ENTSOE_API_BASE', default_var='https://web-api.tp.entsoe.eu'))
    token = os.getenv('ENTSOE_SECURITY_TOKEN', Variable.get('ENTSOE_SECURITY_TOKEN', default_var=''))
    area = os.getenv('ENTSOE_AREA', Variable.get('ENTSOE_AREA', default_var='10Y1001A1001A83F'))
    out_area = os.getenv('ENTSOE_OUT_AREA', Variable.get('ENTSOE_OUT_AREA', default_var=None))
    modes = (os.getenv('ENTSOE_MODES', Variable.get('ENTSOE_MODES', default_var='load,generation')) or '').split(',')
    period_start = os.getenv('ENTSOE_PERIOD_START', Variable.get('ENTSOE_PERIOD_START', default_var=None))
    period_end = os.getenv('ENTSOE_PERIOD_END', Variable.get('ENTSOE_PERIOD_END', default_var=None))
    area_names = _get_json_map('ENTSOE_AREA_NAMES')

    cfg = {
        'source_id': 'entsoe_live',
        'live': True,
        'api_base': api_base,
        'api_token': token,
        'area': area,
        'out_area': out_area,
        'modes': [m.strip() for m in modes if m.strip()],
        'period_start': period_start,
        'period_end': period_end,
        'area_names': area_names,
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }
    connector = ENTSOETransparencyConnector(cfg)
    return connector.run()


def run_entsoe_backfill(**context):
    api_base = os.getenv('ENTSOE_API_BASE', Variable.get('ENTSOE_API_BASE', default_var='https://web-api.tp.entsoe.eu'))
    token = os.getenv('ENTSOE_SECURITY_TOKEN', Variable.get('ENTSOE_SECURITY_TOKEN', default_var=''))
    area = os.getenv('ENTSOE_AREA', Variable.get('ENTSOE_AREA', default_var='10Y1001A1001A83F'))
    out_area = os.getenv('ENTSOE_OUT_AREA', Variable.get('ENTSOE_OUT_AREA', default_var=None))
    modes = (os.getenv('ENTSOE_MODES', Variable.get('ENTSOE_MODES', default_var='load,generation')) or '').split(',')
    area_names = _get_json_map('ENTSOE_AREA_NAMES')

    start = os.getenv('ENTSOE_BACKFILL_START', Variable.get('ENTSOE_BACKFILL_START', default_var=None))
    end = os.getenv('ENTSOE_BACKFILL_END', Variable.get('ENTSOE_BACKFILL_END', default_var=None))
    if not start or not end:
        return 0

    # Chunk windows in hours
    step_hours = int(os.getenv('ENTSOE_BACKFILL_STEP_HOURS', Variable.get('ENTSOE_BACKFILL_STEP_HOURS', default_var='24')))

    def to_dt(s: str) -> datetime:
        # Accept YYYYMMDDHHMM or ISO 'YYYY-MM-DDTHH:MM'
        try:
            return datetime.strptime(s, '%Y%m%d%H%M')
        except Exception:
            return datetime.fromisoformat(s.replace('Z', ''))

    cur = to_dt(start)
    end_dt = to_dt(end)
    total = 0
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(hours=step_hours - 1), end_dt)
        cfg = {
            'source_id': f'entsoe_backfill_{cur:%Y%m%d%H%M}_{chunk_end:%Y%m%d%H%M}',
            'live': True,
            'api_base': api_base,
            'api_token': token,
            'area': area,
            'out_area': out_area,
            'modes': [m.strip() for m in modes if m.strip()],
            'period_start': cur.strftime('%Y%m%d%H%M'),
            'period_end': chunk_end.strftime('%Y%m%d%H%M'),
            'area_names': area_names,
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
        }
        connector = ENTSOETransparencyConnector(cfg)
        total += connector.run()
        cur = chunk_end + timedelta(minutes=1)
    return total


with DAG(
    dag_id='entsoe_ingestion',
    default_args=default_args,
    description='Ingest ENTSO-E fundamentals (load, generation, flows, DA price) to Kafka fundamentals',
    schedule_interval=os.getenv('ENTSOE_SCHEDULE', '0 5 * * *'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    backfill = PythonOperator(
        task_id='backfill_entsoe',
        python_callable=run_entsoe_backfill,
        provide_context=True,
    )
    ingest = PythonOperator(
        task_id='ingest_entsoe',
        python_callable=run_entsoe_live,
        provide_context=True,
    )

    backfill >> ingest

