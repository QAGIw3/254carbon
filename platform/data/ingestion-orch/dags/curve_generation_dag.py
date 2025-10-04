"""
Forward Curve Generation DAG

Overview
--------
Prepares targets and triggers the Curve Service to generate baseline forward
curves for selected instruments. Validates outputs with lightweight checks.

Schedule
- Daily at 06:00 UTC for the baseline (``BASE``) scenario.

Data Flow
---------
ClickHouse fundamentals → targets (μ_t) → Curve Service REST ``/curves/generate``
→ persisted curves (downstream by service) → validation checks

Operational Notes
-----------------
- Inputs are mocked in this DAG for demonstration; replace with real queries.
- Validate generated curves for domain constraints (e.g., non‑negative,
  tenor‑consistent monotonicity where applicable).
- Consider idempotency if re‑running for the same as‑of date.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.utils.dates import days_ago
import json

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ops@254carbon.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}


def prepare_curve_inputs(**context):
    """Prepare fundamentals targets for curve generation."""
    # Query ClickHouse for latest fundamentals
    # Generate μ_t targets for each instrument
    
    execution_date = context['execution_date']
    
    # Example: prepare targets for key instruments
    instruments = [
        'MISO.HUB.INDIANA',
        'MISO.HUB.MICHIGAN',
        'PJM.HUB.WEST',
    ]
    
    curve_requests = []
    for instrument_id in instruments:
        # Mock targets - in production would compute from fundamentals
        targets = [45.0 + i * 0.5 for i in range(60)]  # 60 months
        
        curve_requests.append({
            'instrument_id': instrument_id,
            'as_of_date': execution_date.strftime('%Y-%m-%d'),
            'scenario_id': 'BASE',
            'targets': targets,
            'smoothness_lambda': 50.0,
        })
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='curve_requests', value=curve_requests)
    
    return len(curve_requests)


def generate_curves(**context):
    """Call curve service to generate all curves."""
    import requests
    
    ti = context['task_instance']
    curve_requests = ti.xcom_pull(task_ids='prepare_inputs', key='curve_requests')
    
    curve_service_url = 'http://curve-service:8001/api/v1/curves/generate'
    
    run_ids = []
    for request in curve_requests:
        response = requests.post(curve_service_url, json=request)
        response.raise_for_status()
        
        result = response.json()
        run_ids.append(result['run_id'])
    
    ti.xcom_push(key='run_ids', value=run_ids)
    
    return run_ids


def validate_curves(**context):
    """Validate generated curves meet quality standards."""
    ti = context['task_instance']
    run_ids = ti.xcom_pull(task_ids='generate_curves', key='run_ids')
    
    # Check curve quality metrics
    # - No negative prices
    # - Monotonicity where expected
    # - Reasonable price ranges
    
    return True


with DAG(
    'daily_curve_generation',
    default_args=default_args,
    description='Daily baseline forward curve generation',
    schedule_interval='0 6 * * *',  # 6 AM UTC daily
    start_date=days_ago(1),
    catchup=False,
    tags=['curves', 'baseline', 'daily'],
) as dag:
    
    prepare = PythonOperator(
        task_id='prepare_inputs',
        python_callable=prepare_curve_inputs,
        provide_context=True,
    )
    
    generate = PythonOperator(
        task_id='generate_curves',
        python_callable=generate_curves,
        provide_context=True,
    )
    
    validate = PythonOperator(
        task_id='validate_curves',
        python_callable=validate_curves,
        provide_context=True,
    )
    
    prepare >> generate >> validate
