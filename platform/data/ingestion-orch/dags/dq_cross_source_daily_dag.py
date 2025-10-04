from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "data-quality",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="dq_cross_source_daily",
    default_args=default_args,
    description="Daily cross-source validation",
    schedule_interval="0 3 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["dq", "cross-source"],
) as dag:

    run = BashOperator(
        task_id="run_cross_source",
        bash_command="curl -s -X POST http://data-quality-service:8010/jobs/cross-source",
        env={
            "OPENLINEAGE_URL": "http://marquez:5000",
        },
    )


