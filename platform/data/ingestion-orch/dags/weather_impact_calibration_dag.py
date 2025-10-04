"""Weather impact calibration DAG."""

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
from research_config import load_research_config, weather_mapping
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
        push_to_gateway(PROM_GATEWAY, job="weather_impact_calibration", registry=registry)
    except Exception:  # pragma: no cover
        logger.debug("Prometheus push failed", exc_info=True)


def _weather_entities() -> List[str]:
    config = load_research_config()
    pipeline_cfg: Dict[str, any] = config.get("pipelines", {}).get("weather_impact", {})
    entities = pipeline_cfg.get("entities") or []
    if entities:
        return entities
    inferred = []
    for entity_id, entity_cfg in config.get("entities", {}).items():
        if entity_cfg.get("weather"):
            inferred.append(entity_id)
    return inferred


def calibrate_weather_impact(**context) -> int:
    dag_run = context.get("dag_run")
    start_dt: Optional[datetime] = None
    if dag_run and getattr(dag_run, "conf", None):
        start_conf = dag_run.conf.get("start_date")
        if start_conf:
            start_dt = datetime.fromisoformat(start_conf)

    entities = _weather_entities()
    if not entities:
        logger.warning("No entities configured for weather impact calibration")
        return 0

    data_access = DataAccessLayer()
    persistence = ResearchPersistence(ch_client=data_access.client)
    framework = CommodityResearchFramework(
        data_access=data_access,
        persistence=persistence,
    )

    processed = 0
    for entity_id in entities:
        mapping = weather_mapping(entity_id)
        if not mapping:
            logger.info("Skipping %s: no weather mapping", entity_id)
            continue

        instrument_id = mapping.get("instrument_id")
        if not instrument_id:
            logger.info("Skipping %s: missing instrument mapping", entity_id)
            continue

        lookback = mapping.get("lookback_days", 365)
        if start_dt:
            lookback = max((datetime.utcnow() - start_dt).days, 180)
        window_days = mapping.get("window_days", 90)
        lags = mapping.get("lags")

        temperature_cfg = mapping.get("temperature") or {}
        if not temperature_cfg.get("entity_id") or not temperature_cfg.get("variable"):
            logger.info("Skipping %s: temperature mapping incomplete", entity_id)
            continue

        temperature_series = data_access.get_weather_series(
            temperature_cfg["entity_id"],
            temperature_cfg["variable"],
            start=start_dt,
            lookback_days=lookback,
        )
        if temperature_series.empty or temperature_series.std(skipna=True) == 0:
            logger.info("Skipping %s: no temperature series", entity_id)
            continue

        heating_series = None
        heating_cfg = mapping.get("hdd") or {}
        if heating_cfg.get("entity_id") and heating_cfg.get("variable"):
            heating_series = data_access.get_fundamental_series(
                heating_cfg["entity_id"],
                heating_cfg["variable"],
                start=start_dt,
                lookback_days=lookback,
            )

        cooling_series = None
        cooling_cfg = mapping.get("cdd") or {}
        if cooling_cfg.get("entity_id") and cooling_cfg.get("variable"):
            cooling_series = data_access.get_fundamental_series(
                cooling_cfg["entity_id"],
                cooling_cfg["variable"],
                start=start_dt,
                lookback_days=lookback,
            )

        try:
            prices = data_access.get_price_series(
                instrument_id,
                start=start_dt,
                lookback_days=lookback,
            )
            if prices.empty:
                logger.info("Skipping %s: no price history", entity_id)
                continue

            framework.analyze_weather_impact(
                prices=prices,
                temperature_data=temperature_series,
                heating_demand=heating_series,
                cooling_demand=cooling_series,
                entity_id=entity_id,
                window=f"{window_days}D",
                lags=lags,
                persist=True,
            )
            processed += 1
        except ValueError as exc:
            logger.info("Skipping %s: %s", entity_id, exc)
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("Weather impact failed for %s: %s", entity_id, exc)

    persistence.close()
    data_access.close()
    _emit_metric("research_weather_impact_processed_total", processed)
    return processed


with DAG(
    "weather_impact_calibration",
    default_args=DEFAULT_ARGS,
    description="Weekly weather impact regression calibration",
    schedule_interval="0 5 * * 1",
    start_date=days_ago(1),
    catchup=False,
    tags=["research", "weather"],
) as dag:

    calibrate = PythonOperator(
        task_id="calibrate_weather_impact",
        python_callable=calibrate_weather_impact,
        provide_context=True,
    )
