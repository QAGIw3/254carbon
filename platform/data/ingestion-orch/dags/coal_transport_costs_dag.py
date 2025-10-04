"""Coal transportation economics refresh."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from platform.shared.data_quality_framework import DataQualityFramework

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "supply-chain",
    "email_on_failure": True,
    "email": ["analytics@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=20),
}

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000")
DEFAULT_COAL_ROUTES = os.getenv("COAL_ROUTES", "newcastle_to_rotterdam,richards_bay_to_rotterdam").split(",")
DEFAULT_TRANSPORT_OPTIONS = json.loads(
    os.getenv(
        "COAL_MULTIMODAL_OPTIONS",
        json.dumps(
            {
                "sea": {"cost": 4_500_000, "time_days": 28},
                "rail": {"cost": 2_200_000, "time_days": 12},
                "truck": {"cost": 2_800_000, "time_days": 9},
            }
        ),
    )
)


def _post(path: str, payload: dict) -> dict:
    url = f"{ML_SERVICE_URL}{path}"
    logger.info("Calling ML service: %s", url)
    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def _routes_from_conf(conf: dict | None) -> List[str]:
    if conf and conf.get("routes"):
        routes = conf["routes"]
        if isinstance(routes, list):
            return routes
        if isinstance(routes, str):
            return [item.strip() for item in routes.split(",") if item.strip()]
    return [item.strip() for item in DEFAULT_COAL_ROUTES if item.strip()]


def run_coal_transport(**context) -> int:
    logical_date: datetime = context.get("logical_date") or datetime.utcnow()
    dag_conf = getattr(context.get("dag_run"), "conf", {})
    routes = _routes_from_conf(dag_conf)
    dq = DataQualityFramework()
    rules = dq.get_metric_rules("supply_chain")
    total_records = 0

    for route in routes:
        payload = {
            "as_of_month": logical_date.date().replace(day=1).isoformat(),
            "route": route,
            "cargo_size_tonnes": float(dag_conf.get("cargo_size_tonnes", 75000)),
            "fuel_price_usd_per_tonne": float(dag_conf.get("fuel_price_usd_per_tonne", 620)),
            "include_congestion": dag_conf.get("include_congestion", True),
            "carbon_price_usd_per_tonne": float(dag_conf.get("carbon_price_usd_per_tonne", 50)),
        }
        response = _post("/api/v1/supplychain/coal/route-cost", payload)
        total_records += 1
        total_cost = response.get("total_cost_usd", 0)
        lower = rules["coal_total_route_cost_usd"]["value_min"]
        upper = rules["coal_total_route_cost_usd"]["value_max"]
        if total_cost < lower or total_cost > upper:
            logger.warning(
                "Coal route %s total cost %.0f USD outside [%s, %s]",
                route,
                total_cost,
                lower,
                upper,
            )

        multimodal_payload = {
            "as_of_month": payload["as_of_month"],
            "origin": route.split("_to_")[0],
            "destination": route.split("_to_")[-1],
            "cargo_size_tonnes": payload["cargo_size_tonnes"],
            "transport_options": dag_conf.get("transport_options", DEFAULT_TRANSPORT_OPTIONS),
        }
        multimodal_resp = _post("/api/v1/supplychain/coal/multimodal", multimodal_payload)
        options_frame = pd.DataFrame(multimodal_resp.get("all_options", []))
        if not options_frame.empty:
            negative = options_frame[options_frame["total_cost"] <= 0]
            if not negative.empty:
                logger.warning("Detected non-positive multimodal costs for route %s", route)
        total_records += len(multimodal_resp.get("all_options", []))

    logger.info("Coal transport run generated %s records", total_records)
    return total_records


with DAG(
    dag_id="coal_transport_costs",
    description="Refresh coal transport economics and multimodal trade-offs",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * 1",
    start_date=days_ago(7),
    catchup=False,
    max_active_runs=1,
    tags=["coal", "supply-chain"],
) as dag:
    PythonOperator(
        task_id="compute_coal_transport_costs",
        python_callable=run_coal_transport,
        provide_context=True,
    )
