"""Cross-market arbitrage detection API."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from kafka import KafkaProducer
except Exception:  # pragma: no cover - optional dependency in some environments
    KafkaProducer = None  # type: ignore

try:  # Optional Prometheus metrics
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - optional dependency
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

from data_access import DataAccessLayer
from multi_commodity_portfolio_optimizer import MultiCommodityPortfolioOptimizer
from platform.shared.data_quality_framework import DataQualityFramework

logger = logging.getLogger(__name__)

router = APIRouter()

_data_access = DataAccessLayer()
_optimizer = MultiCommodityPortfolioOptimizer()
_ch_client = _data_access.client

_ARBITRAGE_SIGNAL_TOPIC = os.getenv("ARBITRAGE_SIGNAL_TOPIC", "arbitrage.signals.v1")
_dq = DataQualityFramework()
_STALE_THRESHOLD_DAYS = int(os.getenv("ARBITRAGE_STALE_DAYS", "7"))
_MAX_NET_PROFIT = float(os.getenv("ARBITRAGE_MAX_PROFIT", "1000000"))

try:
    _dq.register_metric_rules(
        "arbitrage",
        {"net_profit": {"value_min": -_MAX_NET_PROFIT, "value_max": _MAX_NET_PROFIT}},
        overwrite=True,
    )
except ValueError:
    logger.debug("Arbitrage metric rules already registered")

if Counter and Histogram:
    _ARBITRAGE_COUNTER = Counter(
        "ml_service_arbitrage_requests_total",
        "Total arbitrage detect requests",
        labelnames=("endpoint", "status"),
    )
    _ARBITRAGE_LATENCY = Histogram(
        "ml_service_arbitrage_latency_seconds",
        "Arbitrage detect latency",
        labelnames=("endpoint",),
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
else:  # pragma: no cover - metrics disabled path
    _ARBITRAGE_COUNTER = None
    _ARBITRAGE_LATENCY = None


def _build_producer() -> Optional[KafkaProducer]:
    if KafkaProducer is None:
        return None
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    try:
        return KafkaProducer(
            bootstrap_servers=bootstrap,
            acks="all",
            enable_idempotence=True,
            compression_type="zstd",
            linger_ms=10,
            value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode("utf-8"),
            key_serializer=lambda value: value.encode("utf-8"),
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("Failed to initialize Kafka producer: %s", exc)
        return None


_producer = _build_producer()


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
                if _ARBITRAGE_COUNTER is not None:
                    _ARBITRAGE_COUNTER.labels(endpoint=endpoint, status=status).inc()
                if _ARBITRAGE_LATENCY is not None:
                    _ARBITRAGE_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)

        return wrapper

    return decorator


class ArbitrageInstrumentSet(BaseModel):
    commodity: str
    instrument_ids: List[str]


class ArbitrageDetectRequest(BaseModel):
    instruments: List[ArbitrageInstrumentSet] = Field(..., min_items=1)
    arbitrage_threshold: float = Field(0.05, gt=0)
    min_correlation: float = Field(0.5, ge=-1, le=1)
    min_sample_count: int = Field(30, ge=0)
    lookback_days: int = Field(180, ge=30, le=1095)
    transport_costs: Optional[Dict[str, Dict[str, float]]] = None
    storage_costs: Optional[Dict[str, float]] = None
    persist: bool = True
    publish_signals: bool = True
    signal_confidence_threshold: float = Field(0.8, ge=0, le=1)


class ArbitrageOpportunityModel(BaseModel):
    instrument1: str
    instrument2: str
    commodity1: str
    commodity2: str
    mean_spread: float
    spread_volatility: float
    transport_cost: float
    storage_cost: float
    net_profit: float
    direction: str
    confidence: float
    period: str
    metadata: Dict[str, Any]


class ArbitrageDetectionResponse(BaseModel):
    as_of_date: datetime
    opportunities: List[ArbitrageOpportunityModel]
    total_opportunities: int


def _serialize_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value).isoformat()
    if isinstance(value, (dict, list, str, float, int)) or value is None:
        return value
    raise TypeError(f"Unsupported type for serialization: {type(value)!r}")


def _load_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@router.post("/api/v1/arbitrage/detect", response_model=ArbitrageDetectionResponse)
@_instrument("arbitrage_detect")
async def detect_arbitrage(request: ArbitrageDetectRequest) -> ArbitrageDetectionResponse:
    instrument_map: Dict[str, str] = {}
    all_instruments: List[str] = []
    for asset in request.instruments:
        for iid in asset.instrument_ids:
            if not iid:
                continue
            instrument_map[iid] = asset.commodity
            all_instruments.append(iid)

    distinct_instruments = sorted(set(all_instruments))
    if len(distinct_instruments) < 2:
        raise HTTPException(status_code=400, detail="At least two instruments are required for arbitrage detection")

    start = datetime.utcnow() - timedelta(days=request.lookback_days)
    prices = _data_access.get_price_dataframe(distinct_instruments, start=start)
    if prices.empty or len(prices.columns) < 2:
        raise HTTPException(status_code=404, detail="Insufficient historical data for requested instruments")

    price_series: Dict[str, pd.Series] = {
        column: series.dropna()
        for column, series in prices.sort_index().items()
        if series.dropna().shape[0] >= 10
    }
    if len(price_series) < 2:
        raise HTTPException(status_code=400, detail="Not enough overlapping history after filtering missing data")

    correlation_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    try:
        corr_rows = _ch_client.execute(
            """
            SELECT instrument1, instrument2, correlation_coefficient, sample_count
            FROM market_intelligence.commodity_correlations
            WHERE date >= today() - 30
              AND instrument1 IN %(instruments)s
              AND instrument2 IN %(instruments)s
            """,
            {"instruments": tuple(price_series.keys())},
        )
    except Exception as exc:
        logger.exception("Failed to retrieve correlation screen")
        raise HTTPException(status_code=500, detail=f"Correlation screening failed: {exc}") from exc

    for instrument1, instrument2, corr_coef, sample_count in corr_rows:
        key = tuple(sorted((instrument1, instrument2)))
        stored = correlation_map.get(key)
        if stored is None or stored.get("sample_count", 0) < sample_count:
            correlation_map[key] = {
                "correlation": float(corr_coef) if corr_coef is not None else None,
                "sample_count": int(sample_count),
            }

    transport_costs = request.transport_costs or {}
    storage_costs = request.storage_costs or {}

    try:
        result = _optimizer.detect_cross_market_arbitrage(
            price_data=price_series,
            transport_costs=transport_costs,
            storage_costs=storage_costs,
            arbitrage_threshold=request.arbitrage_threshold,
        )
    except Exception as exc:
        logger.exception("Arbitrage detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    opportunities: List[ArbitrageOpportunityModel] = []
    inserts: List[List[Any]] = []
    as_of_ts = prices.index.max()
    as_of_date = as_of_ts.to_pydatetime() if isinstance(as_of_ts, pd.Timestamp) else datetime.utcnow()

    for opportunity in result.get("arbitrage_opportunities", []):
        commodity_key1 = opportunity.get("commodity1")
        commodity_key2 = opportunity.get("commodity2")
        pair_key = tuple(sorted((commodity_key1, commodity_key2)))
        correlation_info = correlation_map.get(pair_key)
        if not correlation_info:
            continue
        corr_value = correlation_info.get("correlation")
        sample_count = correlation_info.get("sample_count", 0)
        if corr_value is None or corr_value < request.min_correlation:
            continue
        if sample_count < request.min_sample_count:
            continue

        instrument1 = opportunity.get("commodity1")
        instrument2 = opportunity.get("commodity2")
        commodity1 = instrument_map.get(instrument1, instrument1)
        commodity2 = instrument_map.get(instrument2, instrument2)

        metadata = {
            "correlation": corr_value,
            "sample_count": sample_count,
            "period": opportunity.get("opportunity_period"),
        }
        net_profit = float(opportunity.get("net_profit", 0.0))
        confidence = float(opportunity.get("confidence_score", 0.0))

        if confidence < 0.0 or confidence > 1.0:
            logger.debug("Discarding opportunity %s/%s due to confidence %.2f", instrument1, instrument2, confidence)
            continue
        net_profit_rule = _dq.get_metric_rules("arbitrage").get("net_profit", {})
        lower_bound = float(net_profit_rule.get("value_min", -_MAX_NET_PROFIT))
        upper_bound = float(net_profit_rule.get("value_max", _MAX_NET_PROFIT))
        if not (lower_bound <= net_profit <= upper_bound):
            logger.info(
                "Skipping opportunity %s/%s due to net profit %.2f outside [%s, %s]",
                instrument1,
                instrument2,
                net_profit,
                lower_bound,
                upper_bound,
            )
            continue
        if arbitrage_model.spread_volatility < 0:
            logger.debug("Discarding %s/%s due to negative volatility", instrument1, instrument2)
            continue
        if _STALE_THRESHOLD_DAYS and as_of_date.date() < (datetime.utcnow().date() - timedelta(days=_STALE_THRESHOLD_DAYS)):
            logger.debug("Discarding %s/%s due to stale observation date", instrument1, instrument2)
            continue

        arbitrage_model = ArbitrageOpportunityModel(
            instrument1=instrument1,
            instrument2=instrument2,
            commodity1=commodity1,
            commodity2=commodity2,
            mean_spread=float(opportunity.get("mean_spread", 0.0)),
            spread_volatility=float(opportunity.get("spread_volatility", 0.0)),
            transport_cost=float(opportunity.get("transport_cost", 0.0)),
            storage_cost=float(opportunity.get("storage_cost", 0.0)),
            net_profit=net_profit,
            direction=str(opportunity.get("arbitrage_direction", "")),
            confidence=confidence,
            period=str(opportunity.get("opportunity_period", "")),
            metadata=metadata,
        )
        opportunities.append(arbitrage_model)

        metadata["dq_flags"] = {
            "net_profit_within_bounds": lower_bound <= net_profit <= upper_bound,
            "confidence_valid": 0.0 <= confidence <= 1.0,
            "volatility_non_negative": arbitrage_model.spread_volatility >= 0.0,
        }

        if request.persist:
            inserts.append([
                as_of_date.date(),
                commodity1,
                commodity2,
                instrument1,
                instrument2,
                arbitrage_model.mean_spread,
                arbitrage_model.spread_volatility,
                arbitrage_model.transport_cost,
                arbitrage_model.storage_cost,
                net_profit,
                arbitrage_model.direction,
                confidence,
                arbitrage_model.period,
                json.dumps(metadata, default=_serialize_default),
                datetime.utcnow(),
            ])

        if (
            request.publish_signals
            and _producer is not None
            and confidence >= request.signal_confidence_threshold
        ):
            payload = {
                "instrument1": instrument1,
                "instrument2": instrument2,
                "commodity1": commodity1,
                "commodity2": commodity2,
                "net_profit": net_profit,
                "confidence": confidence,
                "as_of_date": as_of_date.isoformat(),
                "direction": arbitrage_model.direction,
            }
            try:
                _producer.send(_ARBITRAGE_SIGNAL_TOPIC, key=instrument1, value=payload)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to publish arbitrage signal for %s/%s: %s", instrument1, instrument2, exc)

    if request.persist and inserts:
        try:
            _ch_client.execute(
                """
                INSERT INTO market_intelligence.cross_market_arbitrage
                (as_of_date, commodity1, commodity2, instrument1, instrument2, mean_spread, spread_volatility,
                 transport_cost, storage_cost, net_profit, direction, confidence, period, metadata, created_at)
                VALUES
                """,
                inserts,
                types_check=True,
            )
        except Exception as exc:
            logger.exception("Failed to persist arbitrage opportunities")
            raise HTTPException(status_code=500, detail=f"Persistence failed: {exc}") from exc

    return ArbitrageDetectionResponse(
        as_of_date=as_of_date,
        opportunities=opportunities,
        total_opportunities=len(opportunities),
    )
