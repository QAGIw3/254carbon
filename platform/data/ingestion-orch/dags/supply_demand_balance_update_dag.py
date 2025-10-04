"""Supply/demand balance update DAG."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

try:  # Optional Prometheus support
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
except Exception:  # pragma: no cover
    CollectorRegistry = None  # type: ignore
    Gauge = None  # type: ignore
    push_to_gateway = None  # type: ignore

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../apps/ml-service")))

from commodity_research_framework import CommodityResearchFramework
from data_access import DataAccessLayer
from research_config import load_research_config, supply_demand_mapping
from research_persistence import ResearchPersistence


DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["data-engineering@254carbon.ai"],
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

logger = logging.getLogger(__name__)
PROM_GATEWAY = os.getenv("PROM_PUSH_GATEWAY")


def _emit_metric(metric: str, value: float) -> None:
    if not PROM_GATEWAY or Gauge is None or CollectorRegistry is None or push_to_gateway is None:
        return
    try:
        registry = CollectorRegistry()
        gauge = Gauge(metric, "Commodity research metric", registry=registry)
        gauge.set(value)
        push_to_gateway(PROM_GATEWAY, job="supply_demand_balance_update", registry=registry)
    except Exception:  # pragma: no cover
        logger.debug("Prometheus push failed", exc_info=True)


def _instrument_list() -> List[str]:
    config = load_research_config()
    pipeline_cfg: Dict[str, any] = config.get("pipelines", {}).get("supply_demand", {})
    instruments = pipeline_cfg.get("instruments") or []
    if instruments:
        return instruments
    inferred = []
    for instrument_id, inst_cfg in config.get("instruments", {}).items():
        if inst_cfg.get("supply_demand"):
            inferred.append(instrument_id)
    return inferred


def update_supply_demand_balance(**context) -> int:
    dag_run = context.get("dag_run")
    start_dt: Optional[datetime] = None
    if dag_run and getattr(dag_run, "conf", None):
        start_conf = dag_run.conf.get("start_date")
        if start_conf:
            start_dt = datetime.fromisoformat(start_conf)

    instruments = _instrument_list()
    if not instruments:
        logger.warning("No instruments configured for supply/demand update")
        return 0

    data_access = DataAccessLayer()
    persistence = ResearchPersistence(ch_client=data_access.client)
    framework = CommodityResearchFramework(
        data_access=data_access,
        persistence=persistence,
    )

    processed = 0
    for instrument_id in instruments:
        mapping = supply_demand_mapping(instrument_id)
        if not mapping:
            logger.info("Skipping %s: no supply/demand mapping", instrument_id)
            continue

        inventory_cfg = mapping.get("inventory") or {}
        production_cfg = mapping.get("production") or {}
        consumption_cfg = mapping.get("consumption") or {}
        lookback = mapping.get("lookback_days", 365)
        if start_dt:
            lookback = max((datetime.utcnow() - start_dt).days, 180)

        inventory_series = None
        if inventory_cfg.get("entity_id") and inventory_cfg.get("variable"):
            inventory_series = data_access.get_fundamental_series(
                inventory_cfg["entity_id"],
                inventory_cfg["variable"],
                lookback_days=inventory_cfg.get("lookback_days", lookback),
            )
            unit_override = inventory_cfg.get("unit")
            if unit_override:
                inventory_series.attrs["unit"] = unit_override

        production_series = None
        if production_cfg.get("entity_id") and production_cfg.get("variable"):
            production_series = data_access.get_fundamental_series(
                production_cfg["entity_id"],
                production_cfg["variable"],
                lookback_days=production_cfg.get("lookback_days", lookback),
            )
            unit_override = production_cfg.get("unit")
            if unit_override:
                production_series.attrs["unit"] = unit_override

        consumption_series = None
        if consumption_cfg.get("entity_id") and consumption_cfg.get("variable"):
            consumption_series = data_access.get_fundamental_series(
                consumption_cfg["entity_id"],
                consumption_cfg["variable"],
                lookback_days=consumption_cfg.get("lookback_days", lookback),
            )
            unit_override = consumption_cfg.get("unit")
            if unit_override:
                consumption_series.attrs["unit"] = unit_override

        try:
            prices = data_access.get_price_series(
                instrument_id,
                start=start_dt,
                lookback_days=lookback,
            )
            if prices.empty:
                logger.info("Skipping %s: no price history", instrument_id)
                continue

            framework.model_supply_demand_balance(
                prices=prices,
                inventory_data=inventory_series,
                production_data=production_series,
                consumption_data=consumption_series,
                instrument_id=instrument_id,
                entity_id=mapping.get("entity_id") or instrument_id,
                persist=True,
            )
            processed += 1
        except ValueError as exc:
            logger.info("Skipping %s: %s", instrument_id, exc)
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("Supply/demand update failed for %s: %s", instrument_id, exc)

    persistence.close()
    data_access.close()
    _emit_metric("research_supply_demand_processed_total", processed)
    return processed


with DAG(
    "supply_demand_balance_update",
    default_args=DEFAULT_ARGS,
    description="Weekly supply/demand balance metrics",
    schedule_interval="0 4 * * 1",
    start_date=days_ago(1),
    catchup=False,
    tags=["research", "supply-demand"],
) as dag:

    update = PythonOperator(
        task_id="update_supply_demand",
        python_callable=update_supply_demand_balance,
        provide_context=True,
    )
