"""
Automated Backtesting Pipeline DAG

Overview
--------
Runs periodic backtests across active instruments and scenarios to quantify
forecast accuracy. Executes concurrent backtest jobs via the Backtesting
Service and aggregates quality metrics.

Schedule
- See DAG definition; defaults to periodic execution with catchup disabled.

Data Flow
---------
PostgreSQL (instruments + scenarios) → Backtesting Service ``/backtest/run`` →
results via XCom → quality checks (MAPE/WAPE/RMSE) → reporting

Operational Notes
-----------------
- Concurrency: async batch of backtests is bounded by Python process/HTTP limits.
- Quality gates: returns a branch label when thresholds fail.
- Observability: consider persisting metrics to a TSDB for historical trends.
"""
from datetime import datetime, timedelta, date
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago

import sys
import os
import httpx
import asyncio
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

BACKTESTING_SERVICE_URL = os.getenv("BACKTESTING_SERVICE_URL", "http://backtesting-service:8005")


def get_active_instruments(**context):
    """Fetch list of active instruments that need backtesting."""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')

    query = """
    SELECT DISTINCT i.instrument_id, i.market, i.product
    FROM pg.instrument i
    WHERE i.market IN ('MISO', 'CAISO', 'PJM', 'ERCOT', 'SPP')
    ORDER BY i.market, i.product, i.instrument_id
    """

    results = pg_hook.get_records(query)
    instruments = [
        {
            'instrument_id': r[0],
            'market': r[1],
            'product': r[2]
        }
        for r in results
    ]

    context['task_instance'].xcom_push(key='instruments', value=instruments)
    return instruments


def get_scenarios_for_backtesting(**context):
    """Get scenarios that should be backtested."""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')

    query = """
    SELECT scenario_id, title
    FROM pg.scenario
    WHERE scenario_id IN ('BASE', 'HIGH_GAS', 'LOW_GAS', 'HIGH_LOAD', 'LOW_LOAD')
    ORDER BY scenario_id
    """

    results = pg_hook.get_records(query)
    scenarios = [
        {
            'scenario_id': r[0],
            'title': r[1]
        }
        for r in results
    ]

    context['task_instance'].xcom_push(key='scenarios', value=scenarios)
    return scenarios


def run_backtest_for_instrument_and_scenario(**context):
    """Run backtest for a specific instrument and scenario."""
    instruments = context['task_instance'].xcom_pull(task_ids='get_active_instruments', key='instruments')
    scenarios = context['task_instance'].xcom_pull(task_ids='get_scenarios_for_backtesting', key='scenarios')

    # Run backtests for each instrument-scenario combination
    backtest_results = []

    async def run_single_backtest(instrument, scenario):
        """Run a single backtest asynchronously."""
        forecast_date = date.today() - timedelta(days=1)  # Yesterday's forecast
        evaluation_start = forecast_date - timedelta(days=180)  # Last 6 months
        evaluation_end = forecast_date

        request_data = {
            'instrument_id': instrument['instrument_id'],
            'scenario_id': scenario['scenario_id'],
            'forecast_date': forecast_date.isoformat(),
            'evaluation_start': evaluation_start.isoformat(),
            'evaluation_end': evaluation_end.isoformat(),
            'tenor_type': 'Month'
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{BACKTESTING_SERVICE_URL}/api/v1/backtest/run",
                    json=request_data,
                    timeout=30.0
                )

                if response.status_code == 200:
                    result = response.json()
                    result.update({
                        'instrument': instrument,
                        'scenario': scenario,
                        'success': True
                    })
                    return result
                else:
                    result = {
                        'instrument': instrument,
                        'scenario': scenario,
                        'success': False,
                        'error': f"HTTP {response.status_code}: {response.text}"
                    }
                    return result

        except Exception as e:
            result = {
                'instrument': instrument,
                'scenario': scenario,
                'success': False,
                'error': str(e)
            }
            return result

    async def run_all_backtests():
        """Run all backtests concurrently."""
        tasks = []

        for instrument in instruments:
            for scenario in scenarios:
                task = run_single_backtest(instrument, scenario)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and format results
        successful_results = []
        failed_results = []

        for result in results:
            if isinstance(result, Exception):
                print(f"Exception in backtest: {result}")
                continue

            if result['success']:
                successful_results.append(result)
            else:
                failed_results.append(result)

        return successful_results, failed_results

    # Run the async backtests
    successful_results, failed_results = asyncio.run(run_all_backtests())

    # Log results
    print(f"Successful backtests: {len(successful_results)}")
    print(f"Failed backtests: {len(failed_results)}")

    if failed_results:
        print("Failed backtests:")
        for failure in failed_results[:5]:  # Show first 5 failures
            print(f"  {failure['instrument']['instrument_id']} - {failure['scenario']['scenario_id']}: {failure['error']}")

    context['task_instance'].xcom_push(key='successful_backtests', value=len(successful_results))
    context['task_instance'].xcom_push(key='failed_backtests', value=len(failed_results))
    context['task_instance'].xcom_push(key='backtest_results', value=successful_results[:100])  # Limit for XCom

    return len(successful_results)


def check_backtest_quality(**context):
    """Check if backtest results meet quality thresholds."""
    successful = context['task_instance'].xcom_pull(task_ids='run_backtest_for_instrument_and_scenario', key='successful_backtests')
    results = context['task_instance'].xcom_pull(task_ids='run_backtest_for_instrument_and_scenario', key='backtest_results')

    if not results:
        return "quality_check_failed"

    # Calculate aggregate metrics
    mapes = [r['mape'] for r in results if 'mape' in r]
    wapes = [r['wape'] for r in results if 'wape' in r]
    rmses = [r['rmse'] for r in results if 'rmse' in r]

    if not mapes:
        return "quality_check_failed"

    avg_mape = sum(mapes) / len(mapes)
    avg_wape = sum(wapes) / len(wapes) if wapes else 0
    avg_rmse = sum(rmses) / len(rmses) if rmses else 0

    print(f"Average MAPE: {avg_mape".2f"}%")
    print(f"Average WAPE: {avg_wape".2f"}%")
    print(f"Average RMSE: {avg_rmse".2f"}")

    # Quality gates (from requirements)
    mape_threshold = 12.0  # ≤12% for months 1-6

    if avg_mape <= mape_threshold:
        print(f"✅ MAPE quality gate passed: {avg_mape".2f"}% ≤ {mape_threshold}%")
        return "quality_check_passed"
    else:
        print(f"❌ MAPE quality gate failed: {avg_mape".2f"}% > {mape_threshold}%")
        return "quality_check_failed"


def create_grafana_dashboard(**context):
    """Create or update Grafana dashboard with backtest metrics."""
    # This would integrate with Grafana API to create/update dashboards
    # For now, just log the metrics
    results = context['task_instance'].xcom_pull(task_ids='run_backtest_for_instrument_and_scenario', key='backtest_results')

    if results:
        mapes = [r['mape'] for r in results if 'mape' in r]
        wapes = [r['wape'] for r in results if 'wape' in r]
        rmses = [r['rmse'] for r in results if 'rmse' in r]

        avg_mape = sum(mapes) / len(mapes) if mapes else 0
        avg_wape = sum(wapes) / len(wapes) if wapes else 0
        avg_rmse = sum(rmses) / len(rmses) if rmses else 0

        print("📊 Backtest Summary Metrics:")
        print(f"  Average MAPE: {avg_mape".2f"}%")
        print(f"  Average WAPE: {avg_wape".2f"}%")
        print(f"  Average RMSE: {avg_rmse".2f"}")
        print(f"  Total Backtests: {len(results)}")

        # Here you would typically call Grafana API to update dashboard
        # For now, just log that we would update it
        print("📈 Would update Grafana dashboard with these metrics")


def alert_on_poor_performance(**context):
    """Send alerts if backtest performance is poor."""
    results = context['task_instance'].xcom_pull(task_ids='run_backtest_for_instrument_and_scenario', key='backtest_results')

    if not results:
        return

    mapes = [r['mape'] for r in results if 'mape' in r]
    avg_mape = sum(mapes) / len(mapes) if mapes else 0

    # Alert threshold - if average MAPE > 15%
    alert_threshold = 15.0

    if avg_mape > alert_threshold:
        print(f"🚨 ALERT: Average MAPE {avg_mape".2f"}% exceeds threshold {alert_threshold}%")
        # Here you would send actual alerts (Slack, PagerDuty, etc.)
        print("📧 Would send alert email to data-science@254carbon.ai")


# Daily backtesting DAG
with DAG(
    'daily_backtesting',
    default_args=default_args,
    description='Daily automated backtesting for forecast accuracy validation',
    schedule_interval='0 2 * * *',  # Daily at 2 AM UTC
    start_date=days_ago(1),
    catchup=False,
    tags=['backtesting', 'quality', 'daily'],
    max_active_runs=1,
) as dag_daily:

    start_task = DummyOperator(task_id='start')

    get_instruments_task = PythonOperator(
        task_id='get_active_instruments',
        python_callable=get_active_instruments,
        provide_context=True,
    )

    get_scenarios_task = PythonOperator(
        task_id='get_scenarios_for_backtesting',
        python_callable=get_scenarios_for_backtesting,
        provide_context=True,
    )

    run_backtests_task = PythonOperator(
        task_id='run_backtest_for_instrument_and_scenario',
        python_callable=run_backtest_for_instrument_and_scenario,
        provide_context=True,
    )

    quality_check_task = BranchPythonOperator(
        task_id='check_backtest_quality',
        python_callable=check_backtest_quality,
        provide_context=True,
    )

    quality_passed_task = PythonOperator(
        task_id='quality_check_passed',
        python_callable=create_grafana_dashboard,
        provide_context=True,
    )

    quality_failed_task = PythonOperator(
        task_id='quality_check_failed',
        python_callable=alert_on_poor_performance,
        provide_context=True,
    )

    end_task = DummyOperator(task_id='end')

    # Define workflow
    start_task >> [get_instruments_task, get_scenarios_task] >> run_backtests_task >> quality_check_task
    quality_check_task >> quality_passed_task >> end_task
    quality_check_task >> quality_failed_task >> end_task
