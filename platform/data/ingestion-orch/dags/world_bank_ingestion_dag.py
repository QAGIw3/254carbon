"""
World Bank Ingestion DAG

Runs World Bank connectors (energy + economics) in live mode to publish
indicators to Kafka fundamentals for downstream persistence.
"""
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))
from external.infrastructure.world_bank_energy_connector import WorldBankEnergyConnector
from external.economics.world_bank_econ_connector import WorldBankEconomicsConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def run_wb_energy(**context):
    country = os.getenv('WB_COUNTRY', Variable.get('WB_COUNTRY', default_var='WLD'))
    indicators = os.getenv('WB_ENERGY_INDICATORS', Variable.get('WB_ENERGY_INDICATORS', default_var='EG.USE.PCAP.KG.OE,EG.ELC.ACCS.ZS')).split(',')
    start_year = int(os.getenv('WB_START_YEAR', Variable.get('WB_START_YEAR', default_var='2010')))
    end_year = int(os.getenv('WB_END_YEAR', Variable.get('WB_END_YEAR', default_var='2050')))
    cfg = {
        'source_id': 'world_bank_energy_live',
        'live': True,
        'api_base': os.getenv('WB_API_BASE', 'https://api.worldbank.org/v2'),
        'country': country,
        'indicators': indicators,
        'start_year': start_year,
        'end_year': end_year,
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }
    connector = WorldBankEnergyConnector(cfg)
    return connector.run()


def run_wb_econ(**context):
    country = os.getenv('WB_COUNTRY', Variable.get('WB_COUNTRY', default_var='WLD'))
    indicators = os.getenv('WB_ECON_INDICATORS', Variable.get('WB_ECON_INDICATORS', default_var='NY.GDP.PCAP.CD,FP.CPI.TOTL.ZG')).split(',')
    start_year = int(os.getenv('WB_START_YEAR', Variable.get('WB_START_YEAR', default_var='2010')))
    end_year = int(os.getenv('WB_END_YEAR', Variable.get('WB_END_YEAR', default_var='2050')))
    cfg = {
        'source_id': 'world_bank_econ_live',
        'live': True,
        'api_base': os.getenv('WB_API_BASE', 'https://api.worldbank.org/v2'),
        'country': country,
        'indicators': indicators,
        'start_year': start_year,
        'end_year': end_year,
        'kafka_topic': 'market.fundamentals',
        'kafka_bootstrap': os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'),
    }
    connector = WorldBankEconomicsConnector(cfg)
    return connector.run()


def run_wb_backfill_chunked(**context):
    """Backfill World Bank indicators in year chunks to reduce payload size.

    Env inputs:
    - WB_BACKFILL_START_YEAR
    - WB_BACKFILL_END_YEAR
    - WB_BACKFILL_STEP (years per chunk)
    Runs both energy and econ indicators per configured lists.
    """
    start = int(os.getenv('WB_BACKFILL_START_YEAR', Variable.get('WB_BACKFILL_START_YEAR', default_var='2000')))
    end = int(os.getenv('WB_BACKFILL_END_YEAR', Variable.get('WB_BACKFILL_END_YEAR', default_var='2025')))
    step = int(os.getenv('WB_BACKFILL_STEP', Variable.get('WB_BACKFILL_STEP', default_var='10')))
    country = os.getenv('WB_COUNTRY', Variable.get('WB_COUNTRY', default_var='WLD'))
    energy_inds = os.getenv('WB_ENERGY_INDICATORS', Variable.get('WB_ENERGY_INDICATORS', default_var='EG.USE.PCAP.KG.OE,EG.ELC.ACCS.ZS')).split(',')
    econ_inds = os.getenv('WB_ECON_INDICATORS', Variable.get('WB_ECON_INDICATORS', default_var='NY.GDP.PCAP.CD,FP.CPI.TOTL.ZG')).split(',')
    api_base = os.getenv('WB_API_BASE', Variable.get('WB_API_BASE', default_var='https://api.worldbank.org/v2'))
    kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP', Variable.get('KAFKA_BOOTSTRAP', default_var='kafka:9092'))

    total = 0
    for chunk_start in range(start, end + 1, step):
        chunk_end = min(chunk_start + step - 1, end)
        # energy
        cfg_energy = {
            'source_id': f'world_bank_energy_backfill_{chunk_start}_{chunk_end}',
            'live': True,
            'api_base': api_base,
            'country': country,
            'indicators': energy_inds,
            'start_year': chunk_start,
            'end_year': chunk_end,
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': kafka_bootstrap,
        }
        total += WorldBankEnergyConnector(cfg_energy).run()

        # econ
        cfg_econ = {
            'source_id': f'world_bank_econ_backfill_{chunk_start}_{chunk_end}',
            'live': True,
            'api_base': api_base,
            'country': country,
            'indicators': econ_inds,
            'start_year': chunk_start,
            'end_year': chunk_end,
            'kafka_topic': 'market.fundamentals',
            'kafka_bootstrap': kafka_bootstrap,
        }
        total += WorldBankEconomicsConnector(cfg_econ).run()

    return total


with DAG(
    dag_id='world_bank_ingestion',
    default_args=default_args,
    description='Ingest World Bank energy + economics indicators to Kafka fundamentals',
    schedule_interval=os.getenv('WB_SCHEDULE', '@weekly'),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    ingest_energy = PythonOperator(
        task_id='ingest_world_bank_energy',
        python_callable=run_wb_energy,
        provide_context=True,
    )
    ingest_econ = PythonOperator(
        task_id='ingest_world_bank_econ',
        python_callable=run_wb_econ,
        provide_context=True,
    )
    backfill = PythonOperator(
        task_id='backfill_world_bank',
        python_callable=run_wb_backfill_chunked,
        provide_context=True,
    )

    backfill >> ingest_energy >> ingest_econ
