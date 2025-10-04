from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "data-quality",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="dq_imputation_daily",
    default_args=default_args,
    description="Daily imputation",
    schedule_interval="30 3 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["dq", "imputation"],
) as dag:

    run = BashOperator(
        task_id="run_imputation",
        bash_command="curl -s -X POST http://data-quality-service:8010/jobs/imputation",
        env={
            "OPENLINEAGE_URL": "http://marquez:5000",
        },
    )


