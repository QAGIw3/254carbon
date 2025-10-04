"""Gas & Coal Analytics Service."""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

from .analytics.storage_arbitrage import StorageArbitrageCalculator
from .analytics.weather_impact import WeatherImpactAnalyzer, DEFAULT_WINDOW
from .analytics.coal_to_gas import CoalToGasSwitchingCalculator
from .analytics.gas_basis import GasBasisModeler
from .clients.clickhouse import query_dataframe
from .models import (
    StorageArbitrageResult,
    StorageArbitrageScheduleEntry,
    WeatherImpactCoefficient,
    CoalToGasSwitchingResult,
    GasBasisModelResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gas & Coal Analytics",
    description="Storage arbitrage, weather impact, switching economics, and basis modeling",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _load_storage_from_db(hub: str, as_of: date) -> Optional[StorageArbitrageResult]:
    sql = """
        SELECT
            as_of_date,
            hub,
            region,
            expected_storage_value,
            breakeven_spread,
            optimal_schedule,
            cost_parameters,
            constraint_summary,
            diagnostics
        FROM ch.gas_storage_arbitrage
        WHERE hub = %(hub)s
          AND as_of_date = %(as_of)s
        ORDER BY created_at DESC
        LIMIT 1
    """
    df = query_dataframe(sql, {"hub": hub, "as_of": as_of})
    if df.empty:
        return None
    row = df.iloc[0]
    schedule = [StorageArbitrageScheduleEntry(**item) for item in json.loads(row["optimal_schedule"]) ]
    return StorageArbitrageResult(
        as_of_date=row["as_of_date"],
        hub=row["hub"],
        region=row["region"],
        expected_storage_value=row["expected_storage_value"],
        breakeven_spread=row["breakeven_spread"],
        schedule=schedule,
        cost_parameters=json.loads(row["cost_parameters"]),
        constraint_summary=json.loads(row["constraint_summary"]),
        diagnostics=json.loads(row["diagnostics"]),
    )


@app.get("/api/v1/gas/arbitrage", response_model=StorageArbitrageResult)
def get_storage_arbitrage(
    hub: str = Query(..., description="Storage hub identifier"),
    as_of: date = Query(..., description="As-of date"),
    refresh: bool = Query(False, description="Force recomputation"),
    persist: bool = Query(False, description="Persist recomputed result to ClickHouse"),
) -> StorageArbitrageResult:
    if not refresh:
        cached = _load_storage_from_db(hub.upper(), as_of)
        if cached:
            return cached
    calculator = StorageArbitrageCalculator()
    result = calculator.compute(hub.upper(), as_of)
    if persist:
        calculator.persist(result)
    return result


def _load_weather_from_db(entity: str, window_label: str, limit: int = 10) -> List[WeatherImpactCoefficient]:
    sql = """
        SELECT
            date,
            entity_id,
            coef_type,
            coefficient,
            r2,
            window,
            diagnostics
        FROM ch.weather_impact
        WHERE entity_id = %(entity)s
          AND window = %(window)s
        ORDER BY date DESC
        LIMIT %(limit)s
    """
    df = query_dataframe(sql, {"entity": entity, "window": window_label, "limit": limit})
    outputs: List[WeatherImpactCoefficient] = []
    for _, row in df.iterrows():
        outputs.append(
            WeatherImpactCoefficient(
                as_of_date=row["date"],
                entity_id=row["entity_id"],
                coef_type=row["coef_type"],
                coefficient=row["coefficient"],
                r2=row["r2"],
                window=row["window"],
                diagnostics=json.loads(row["diagnostics"]) if row["diagnostics"] else {},
            )
        )
    return outputs


@app.get("/api/v1/analytics/weather-impact", response_model=List[WeatherImpactCoefficient])
def get_weather_impact(
    entity: str = Query(..., description="Hub or region identifier"),
    as_of: date = Query(..., description="As-of date for regression window"),
    window: int = Query(DEFAULT_WINDOW, description="Rolling window in days"),
    refresh: bool = Query(False, description="Force recomputation"),
    persist: bool = Query(False, description="Persist computed coefficients"),
) -> List[WeatherImpactCoefficient]:
    window_label = f"{window}d"
    if not refresh:
        cached = _load_weather_from_db(entity.upper(), window_label)
        if cached:
            return cached
    analyzer = WeatherImpactAnalyzer(window=window)
    coeffs = analyzer.run(entity.upper(), as_of)
    if persist:
        analyzer.persist(coeffs)
    return coeffs


def _load_ctg_from_db(region: str, as_of: date) -> Optional[CoalToGasSwitchingResult]:
    sql = """
        SELECT
            as_of_date,
            region,
            coal_cost_mwh,
            gas_cost_mwh,
            co2_price,
            breakeven_gas_price,
            switch_share,
            diagnostics
        FROM ch.coal_gas_switching
        WHERE region = %(region)s
          AND as_of_date = %(as_of)s
        ORDER BY created_at DESC
        LIMIT 1
    """
    df = query_dataframe(sql, {"region": region, "as_of": as_of})
    if df.empty:
        return None
    row = df.iloc[0]
    return CoalToGasSwitchingResult(
        as_of_date=row["as_of_date"],
        region=row["region"],
        coal_cost_mwh=row["coal_cost_mwh"],
        gas_cost_mwh=row["gas_cost_mwh"],
        co2_price=row["co2_price"],
        breakeven_gas_price=row["breakeven_gas_price"],
        switch_share=row["switch_share"],
        diagnostics=json.loads(row["diagnostics"]) if row["diagnostics"] else {},
    )


@app.get("/api/v1/analytics/ctg-switch", response_model=CoalToGasSwitchingResult)
def get_coal_to_gas_switch(
    region: str = Query(..., description="Regional identifier"),
    as_of: date = Query(..., description="As-of date"),
    co2_price: Optional[float] = Query(None, description="Override for CO2 price"),
    refresh: bool = Query(False, description="Force recomputation"),
    persist: bool = Query(False, description="Persist computed results"),
) -> CoalToGasSwitchingResult:
    if not refresh:
        cached = _load_ctg_from_db(region.upper(), as_of)
        if cached:
            return cached
    calculator = CoalToGasSwitchingCalculator()
    result = calculator.compute(region.upper(), as_of, co2_price=co2_price)
    if persist:
        calculator.persist(result)
    return result


def _load_basis_from_db(hub: str, as_of: date) -> Optional[GasBasisModelResult]:
    sql = """
        SELECT
            as_of_date,
            hub,
            predicted_basis,
            actual_basis,
            feature_snapshot,
            diagnostics,
            method
        FROM ch.gas_basis_models
        WHERE hub = %(hub)s
          AND as_of_date = %(as_of)s
        ORDER BY created_at DESC
        LIMIT 1
    """
    df = query_dataframe(sql, {"hub": hub, "as_of": as_of})
    if df.empty:
        return None
    row = df.iloc[0]
    return GasBasisModelResult(
        as_of_date=row["as_of_date"],
        hub=row["hub"],
        predicted_basis=row["predicted_basis"],
        actual_basis=row["actual_basis"],
        method=row["method"],
        diagnostics=json.loads(row["diagnostics"]),
        feature_snapshot=json.loads(row["feature_snapshot"]),
    )


@app.get("/api/v1/gas/basis/model", response_model=GasBasisModelResult)
def get_gas_basis_model(
    hub: str = Query(..., description="Hub identifier"),
    as_of: date = Query(..., description="As-of date"),
    refresh: bool = Query(False, description="Force recomputation"),
    persist: bool = Query(False, description="Persist computed output"),
) -> GasBasisModelResult:
    if not refresh:
        cached = _load_basis_from_db(hub.upper(), as_of)
        if cached:
            return cached
    modeler = GasBasisModeler()
    try:
        result = modeler.compute(hub.upper(), as_of)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if persist:
        modeler.persist(result)
    return result


@app.post("/api/v1/jobs/run-daily")
def run_daily_jobs(
    as_of: date = Query(..., description="As-of date for all analytics"),
    hubs: Optional[List[str]] = Query(None, description="Hubs for gas analytics"),
    regions: Optional[List[str]] = Query(None, description="Regions for switching"),
    window: int = Query(DEFAULT_WINDOW, description="Weather regression window"),
) -> dict:
    """Trigger all analytics modules sequentially."""
    from .jobs import (
        run_storage_arbitrage_job,
        run_weather_impact_job,
        run_coal_to_gas_switch_job,
        run_basis_model_job,
        run_hdd_cdd_metrics_job,
    )

    hubs_list = hubs or ["HENRY", "DAWN"]
    regions_list = regions or ["PJM", "ERCOT"]

    metric_entities = sorted({entity.upper() for entity in (list(hubs_list) + list(regions_list))})
    metrics_rows = run_hdd_cdd_metrics_job(as_of, metric_entities)
    storage_results = run_storage_arbitrage_job(as_of, hubs_list)
    weather_results = run_weather_impact_job(as_of, hubs_list, window=window)
    switching_results = run_coal_to_gas_switch_job(as_of, regions_list)
    basis_results = run_basis_model_job(as_of, hubs_list)

    return {
        "weather_metrics": len(metrics_rows),
        "storage": len(storage_results),
        "weather": len(weather_results),
        "switching": len(switching_results),
        "basis": len(basis_results),
    }
