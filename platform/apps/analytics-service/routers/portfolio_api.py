"""Portfolio optimization API for multi-commodity strategies."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:  # Optional Prometheus metrics
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - metrics optional
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

from data_access import DataAccessLayer
from multi_commodity_portfolio_optimizer import MultiCommodityPortfolioOptimizer

logger = logging.getLogger(__name__)

router = APIRouter()

_data_access = DataAccessLayer()
_optimizer = MultiCommodityPortfolioOptimizer()
_ch_client = _data_access.client

if Counter and Histogram:
    _PORTFOLIO_COUNTER = Counter(
        "ml_service_portfolio_requests_total",
        "Total portfolio API requests",
        labelnames=("endpoint", "status"),
    )
    _PORTFOLIO_LATENCY = Histogram(
        "ml_service_portfolio_latency_seconds",
        "Portfolio API latency",
        labelnames=("endpoint",),
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )
else:  # pragma: no cover - metrics disabled
    _PORTFOLIO_COUNTER = None
    _PORTFOLIO_LATENCY = None


class PortfolioConstraints(BaseModel):
    max_weight: float = Field(0.3, gt=0, le=1)
    target_return: Optional[float] = None


class PortfolioOptimizeRequest(BaseModel):
    instrument_ids: List[str] = Field(..., min_items=2, description="Instruments to include in the optimization")
    portfolio_id: Optional[str] = Field(None, description="Logical portfolio identifier")
    method: str = Field(
        "mean_variance",
        pattern=r"^(mean_variance|risk_parity|equal)$",
        description="Optimization method",
    )
    constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)
    risk_free_rate: float = Field(0.02, description="Risk-free rate used for Sharpe computations")
    lookback_days: int = Field(365, ge=60, le=1825, description="Lookback window for price history")
    persist: bool = Field(True, description="Persist optimization results to ClickHouse")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Auxiliary context for run provenance")


class PortfolioOptimizeResponse(BaseModel):
    run_id: str
    portfolio_id: str
    as_of_date: datetime
    method: str
    weights: Dict[str, float]
    portfolio_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    params: Dict[str, Any]


class PortfolioRunReport(BaseModel):
    run_id: str
    portfolio_id: str
    as_of_date: datetime
    method: str
    params: Dict[str, Any]
    weights: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Series):
        return {k: _json_default(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        frame_dict = value.to_dict()
        return {k: {kk: _json_default(vv) for kk, vv in inner.items()} for k, inner in frame_dict.items()}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list, str, float, int)) or value is None:
        return value
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def _instrument(endpoint: str):
    def decorator(func):
        if Counter is None or Histogram is None:
            return func

        @wraps(func)
        async def wrapper(*args, **kwargs):
            status = "success"
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            except Exception:
                status = "error"
                raise
            finally:
                if _PORTFOLIO_COUNTER is not None:
                    _PORTFOLIO_COUNTER.labels(endpoint=endpoint, status=status).inc()
                if _PORTFOLIO_LATENCY is not None:
                    _PORTFOLIO_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)

        return wrapper

    return decorator


@router.post("/api/v1/portfolio/optimize", response_model=PortfolioOptimizeResponse)
@_instrument("portfolio_optimize")
async def optimize_portfolio(request: PortfolioOptimizeRequest) -> PortfolioOptimizeResponse:
    distinct_instruments = sorted({iid for iid in request.instrument_ids if iid})
    if len(distinct_instruments) < 2:
        raise HTTPException(status_code=400, detail="At least two distinct instruments are required")

    start = datetime.utcnow() - timedelta(days=request.lookback_days)
    prices = _data_access.get_price_dataframe(distinct_instruments, start=start)
    if prices.empty or len(prices.columns) < 2:
        raise HTTPException(status_code=404, detail="Insufficient price history for requested instruments")

    returns = prices.sort_index().pct_change().dropna(how="all")
    if returns.empty:
        raise HTTPException(status_code=400, detail="Unable to compute returns from supplied price history")

    try:
        optimization_result = _optimizer.optimize_portfolio_weights(
            returns_data=returns,
            risk_free_rate=request.risk_free_rate,
            target_return=request.constraints.target_return,
            max_weight=request.constraints.max_weight,
            optimization_method=request.method,
        )
    except Exception as exc:
        logger.exception("Portfolio optimization failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    weights = {instrument: float(weight) for instrument, weight in optimization_result["optimal_weights"].items()}
    portfolio_metrics = {key: float(value) for key, value in optimization_result["portfolio_metrics"].items()}

    try:
        risk_metrics = {
            key: float(val)
            for key, val in _optimizer.calculate_integrated_risk_metrics(weights, returns).items()
        }
    except Exception as exc:
        logger.exception("Integrated risk metric computation failed")
        raise HTTPException(status_code=500, detail=f"Risk metric computation failed: {exc}") from exc

    expected_returns = {
        instrument: float(val) for instrument, val in optimization_result["expected_returns"].items()
    }
    covariance_matrix = {
        row: {col: float(val) for col, val in col_dict.items()}
        for row, col_dict in optimization_result["covariance_matrix"].to_dict().items()
    }

    now_ts = datetime.utcnow()
    as_of_date = prices.index.max().to_pydatetime() if isinstance(prices.index.max(), pd.Timestamp) else now_ts
    run_id = str(uuid4())
    portfolio_id = request.portfolio_id or f"portfolio_{run_id[:8]}"

    params_payload: Dict[str, Any] = {
        "constraints": request.constraints.model_dump(),
        "risk_free_rate": request.risk_free_rate,
        "metadata": request.metadata or {},
        "lookback_days": request.lookback_days,
    }
    metrics_payload: Dict[str, Any] = {
        "portfolio": portfolio_metrics,
        "risk": risk_metrics,
        "expected_returns": expected_returns,
        "covariance_matrix": covariance_matrix,
    }

    if request.persist:
        try:
            _ch_client.execute(
                """
                INSERT INTO market_intelligence.portfolio_optimization_runs
                (run_id, portfolio_id, as_of_date, method, params, weights, metrics, created_at, updated_at)
                VALUES
                """,
                [[
                    run_id,
                    portfolio_id,
                    as_of_date.date(),
                    request.method,
                    json.dumps(params_payload, default=_json_default),
                    json.dumps(weights, default=_json_default),
                    json.dumps(metrics_payload, default=_json_default),
                    now_ts,
                    now_ts,
                ]],
                types_check=True,
            )
        except Exception as exc:
            logger.exception("Failed to persist portfolio optimization run")
            raise HTTPException(status_code=500, detail=f"Persistence failed: {exc}") from exc

    return PortfolioOptimizeResponse(
        run_id=run_id,
        portfolio_id=portfolio_id,
        as_of_date=as_of_date,
        method=request.method,
        weights=weights,
        portfolio_metrics=portfolio_metrics,
        risk_metrics=risk_metrics,
        params=params_payload,
    )


@router.get("/api/v1/portfolio/report", response_model=PortfolioRunReport)
@_instrument("portfolio_report")
async def get_portfolio_run(run_id: str = Query(..., description="Portfolio optimization run identifier")) -> PortfolioRunReport:
    try:
        rows = _ch_client.execute(
            """
            SELECT run_id, portfolio_id, as_of_date, method, params, weights, metrics, created_at, updated_at
            FROM market_intelligence.portfolio_optimization_runs
            WHERE run_id = %(run_id)s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            {"run_id": run_id},
        )
    except Exception as exc:
        logger.exception("Failed to load portfolio run")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    if not rows:
        raise HTTPException(status_code=404, detail=f"No portfolio optimization run found for {run_id}")

    row = rows[0]
    _, portfolio_id, as_of_date, method, params_json, weights_json, metrics_json, created_at, updated_at = row

    if isinstance(as_of_date, datetime):
        as_of_dt = as_of_date
    else:
        as_of_dt = datetime.combine(as_of_date, datetime.min.time())

    try:
        params = json.loads(params_json) if isinstance(params_json, str) else params_json
    except json.JSONDecodeError:
        params = {}
    try:
        weights = json.loads(weights_json) if isinstance(weights_json, str) else weights_json
    except json.JSONDecodeError:
        weights = {}
    try:
        metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
    except json.JSONDecodeError:
        metrics = {}

    return PortfolioRunReport(
        run_id=run_id,
        portfolio_id=portfolio_id,
        as_of_date=as_of_dt,
        method=method,
        params=params or {},
        weights=weights or {},
        metrics=metrics or {},
        created_at=created_at,
        updated_at=updated_at,
    )
