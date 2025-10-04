"""
Risk Analytics Service
Value at Risk (VaR), Expected Shortfall, and portfolio risk aggregation.
"""
import json
import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy import stats

# Use absolute imports to allow running as a module entrypoint
from var_calculator import VaRCalculator
from portfolio import PortfolioAggregator
from stress_testing import StressTestEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Risk Analytics Service",
    description="Portfolio risk metrics and stress testing",
    version="1.0.0",
)

# Initialize components
var_calculator = VaRCalculator()
portfolio_aggregator = PortfolioAggregator()
stress_engine = StressTestEngine()


class Position(BaseModel):
    """Trading position."""
    instrument_id: str
    quantity: float  # MW or contracts
    entry_price: Optional[float] = None


class VaRRequest(BaseModel):
    """VaR calculation request."""
    positions: List[Position]
    confidence_level: float = 0.95  # 95% confidence
    horizon_days: int = 1  # 1-day VaR
    method: str = "historical"  # historical, parametric, monte_carlo


class VaRResponse(BaseModel):
    """VaR calculation response."""
    var_value: float
    expected_shortfall: float
    confidence_level: float
    horizon_days: int
    method: str
    portfolio_value: float
    positions_count: int


class StressTestRequest(BaseModel):
    """Stress test request."""
    positions: List[Position]
    scenarios: List[Dict[str, Any]]


class StressTestResponse(BaseModel):
    """Stress test response."""
    scenario_results: List[Dict[str, Any]]
    worst_case_loss: float
    best_case_gain: float


class RiskMetricsRequest(BaseModel):
    """Portfolio risk metrics request."""

    positions: Optional[List[Position]] = Field(None, description="Explicit positions to analyse")
    portfolio_id: Optional[str] = Field(None, description="Portfolio identifier (if referencing stored optimization runs)")
    run_id: Optional[str] = Field(None, description="Optimization run identifier")
    horizon_days: int = Field(1, ge=1, le=30, description="Risk horizon in trading days")
    confidence: float = Field(0.95, gt=0, lt=1, description="Confidence level for VaR/ES")
    methods: List[str] = Field(default_factory=lambda: ["historical", "parametric", "monte_carlo"], description="Risk methodologies to compute")
    lookback_days: int = Field(252, ge=60, le=1825, description="Historical window for price series")
    persist: bool = Field(True, description="Persist computed metrics to ClickHouse")


class RiskMetricsResponse(BaseModel):
    """Portfolio risk metrics response."""

    portfolio_id: str
    run_id: str
    as_of_date: datetime
    horizon_days: int
    confidence: float
    methods: Dict[str, Dict[str, float]]
    var_95: float
    cvar_95: float
    volatility: float
    variance: float
    max_drawdown: float
    diversification_benefit: float
    beta: Optional[float]
    exposures: Dict[str, Any]


class StressScenario(BaseModel):
    """Stress scenario definition."""

    scenario_id: str = Field(..., alias="id", description="Unique scenario identifier")
    type: str = Field("price_shock", description="Scenario type consumed by the stress engine")
    name: Optional[str] = Field(None, description="Human readable label")
    probability: Optional[float] = Field(None, ge=0, le=1, description="Scenario probability, if calibrated")
    severity: Optional[str] = Field(None, description="Optional severity override")

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class StressPortfolioRequest(BaseModel):
    """Portfolio stress testing request."""

    positions: Optional[List[Position]] = Field(None, description="Explicit positions to stress")
    portfolio_id: Optional[str] = Field(None, description="Portfolio identifier")
    run_id: Optional[str] = Field(None, description="Optimization run identifier")
    scenarios: List[StressScenario] = Field(..., min_items=1)
    lookback_days: int = Field(252, ge=60, le=1825, description="Historical window used to derive latest prices")
    persist: bool = Field(True, description="Persist results to ClickHouse")


class StressScenarioResult(BaseModel):
    """Stress scenario result payload."""

    scenario_id: str
    name: Optional[str]
    probability: float
    severity: str
    pnl: float
    pnl_pct: float
    details: Dict[str, Any]


class StressPortfolioResponse(BaseModel):
    """Portfolio stress testing response."""

    portfolio_id: str
    run_id: str
    as_of_date: datetime
    results: List[StressScenarioResult]


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list, str, float, int)) or value is None:
        return value
    raise TypeError(f"Unsupported type for serialization: {type(value)!r}")


def _load_optional_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _fetch_portfolio_run(portfolio_id: Optional[str], run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    filters: List[str] = []
    params: Dict[str, Any] = {}
    if run_id:
        filters.append("run_id = %(run_id)s")
        params["run_id"] = run_id
    if portfolio_id:
        filters.append("portfolio_id = %(portfolio_id)s")
        params["portfolio_id"] = portfolio_id
    if not filters:
        return None

    query = f"""
        SELECT run_id, portfolio_id, weights, metrics, params
        FROM market_intelligence.portfolio_optimization_runs
        WHERE {' AND '.join(filters)}
        ORDER BY updated_at DESC
        LIMIT 1
    """

    rows = var_calculator.ch_client.execute(query, params)
    if not rows:
        return None

    run_uuid, portfolio_identifier, weights_json, metrics_json, params_json = rows[0]

    weights_raw = _load_optional_json(weights_json) or {}
    if not isinstance(weights_raw, dict):
        weights_raw = {}

    weights = {instrument: float(value) for instrument, value in weights_raw.items() if value is not None}

    return {
        "run_id": str(run_uuid),
        "portfolio_id": portfolio_identifier,
        "weights": weights,
        "metrics": _load_optional_json(metrics_json),
        "params": _load_optional_json(params_json),
    }


def _derive_weights(positions: List[Position], latest_prices: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    valuations: Dict[str, float] = {}
    total_value = 0.0
    for pos in positions:
        price = float(latest_prices.get(pos.instrument_id, pos.entry_price or 0.0) or 0.0)
        if math.isclose(price, 0.0):
            continue
        value = float(pos.quantity) * price
        valuations[pos.instrument_id] = value
        total_value += value

    if math.isclose(total_value, 0.0):
        return {}, {}

    weights = {instrument: value / total_value for instrument, value in valuations.items()}
    return weights, valuations


def _positions_from_weights(weights: Dict[str, float], latest_prices: pd.Series) -> List[Position]:
    positions: List[Position] = []
    for instrument, weight in weights.items():
        price = float(latest_prices.get(instrument, 0.0) or 0.0)
        if math.isclose(price, 0.0):
            price = 1.0
        quantity = weight / price
        positions.append(
            Position(
                instrument_id=instrument,
                quantity=float(quantity),
                entry_price=price,
            )
        )
    return positions


async def _build_portfolio_context(
    positions: Optional[List[Position]],
    portfolio_id: Optional[str],
    run_id: Optional[str],
    lookback_days: int,
) -> Dict[str, Any]:
    run_record = _fetch_portfolio_run(portfolio_id, run_id)
    resolved_positions: List[Position] = []
    resolved_portfolio_id: str
    resolved_run_id: str

    if positions:
        resolved_positions = [
            Position(
                instrument_id=pos.instrument_id,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
            )
            for pos in positions
        ]
        resolved_portfolio_id = portfolio_id or f"ad_hoc_{uuid4().hex[:8]}"
        resolved_run_id = run_id or str(uuid4())
        instrument_ids = sorted({pos.instrument_id for pos in resolved_positions if pos.instrument_id})
    elif run_record:
        resolved_portfolio_id = run_record["portfolio_id"]
        resolved_run_id = run_record["run_id"]
        instrument_ids = sorted(run_record["weights"].keys())
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either positions or a portfolio_id/run_id reference",
        )

    if not instrument_ids:
        raise HTTPException(status_code=400, detail="Resolved instrument universe is empty")

    prices = await var_calculator.get_historical_prices(instrument_ids, lookback_days=lookback_days)
    if prices.empty:
        raise HTTPException(status_code=404, detail="No price history available for resolved instruments")

    cleaned_prices = prices.sort_index().ffill().dropna(how="all")
    if cleaned_prices.empty:
        raise HTTPException(status_code=404, detail="Price history empty after cleaning missing observations")

    latest_prices = cleaned_prices.iloc[-1]

    if not resolved_positions and run_record:
        resolved_positions = _positions_from_weights(run_record["weights"], latest_prices)
    else:
        enriched_positions: List[Position] = []
        for pos in resolved_positions:
            price = float(latest_prices.get(pos.instrument_id, pos.entry_price or 0.0) or 0.0)
            entry_price = pos.entry_price if pos.entry_price is not None else (price if price > 0 else None)
            enriched_positions.append(
                Position(
                    instrument_id=pos.instrument_id,
                    quantity=pos.quantity,
                    entry_price=entry_price,
                )
            )
        resolved_positions = enriched_positions

    weights, valuations = _derive_weights(resolved_positions, latest_prices)
    if not weights:
        raise HTTPException(status_code=400, detail="Unable to derive portfolio weights from supplied inputs")

    context: Dict[str, Any] = {
        "positions": resolved_positions,
        "weights": weights,
        "valuations": valuations,
        "portfolio_id": resolved_portfolio_id,
        "run_id": resolved_run_id,
        "prices": cleaned_prices,
        "latest_prices": latest_prices,
        "as_of_date": cleaned_prices.index.max(),
    }

    if run_record:
        context["source_run"] = run_record

    return context


def _infer_severity(pnl_pct: float) -> str:
    if pnl_pct <= -20:
        return "critical"
    if pnl_pct <= -10:
        return "high"
    if pnl_pct <= -5:
        return "medium"
    return "low"


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/risk/metrics", response_model=RiskMetricsResponse)
async def calculate_portfolio_metrics(request: RiskMetricsRequest) -> RiskMetricsResponse:
    methods = request.methods or ["historical"]
    normalized_methods = [method.lower() for method in methods]

    context = await _build_portfolio_context(
        positions=request.positions,
        portfolio_id=request.portfolio_id,
        run_id=request.run_id,
        lookback_days=request.lookback_days,
    )

    prices = context["prices"]
    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        raise HTTPException(status_code=400, detail="Insufficient return history for risk computation")

    portfolio_returns = var_calculator.build_portfolio_returns(context["positions"], prices)
    if portfolio_returns.empty:
        raise HTTPException(status_code=400, detail="Unable to construct portfolio returns for risk aggregation")

    method_results: Dict[str, Dict[str, float]] = {}

    for method_key in normalized_methods:
        if method_key == "historical":
            var_value, es_value = var_calculator.historical_var(
                portfolio_returns,
                confidence_level=request.confidence,
                horizon_days=request.horizon_days,
            )
        elif method_key == "parametric":
            var_value, es_value = var_calculator.parametric_var(
                portfolio_returns,
                confidence_level=request.confidence,
                horizon_days=request.horizon_days,
            )
        elif method_key in {"monte_carlo", "montecarlo"}:
            var_value, es_value = var_calculator.monte_carlo_var(
                portfolio_returns,
                confidence_level=request.confidence,
                horizon_days=request.horizon_days,
            )
            method_key = "monte_carlo"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported risk method: {method_key}")

        method_results[method_key] = {
            "var": float(var_value),
            "expected_shortfall": float(es_value),
        }

    primary_method_key = normalized_methods[0] if normalized_methods else "historical"
    primary_metrics = method_results.get(primary_method_key) or next(iter(method_results.values()))

    volatility = float(portfolio_returns.std(ddof=1)) if not portfolio_returns.empty else 0.0
    variance = float(portfolio_returns.var(ddof=1)) if not portfolio_returns.empty else 0.0

    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    weights = context["weights"]
    individual_risks = [
        weights[inst] * returns[inst].std(ddof=1)
        for inst in weights.keys()
        if inst in returns and not returns[inst].empty
    ]
    total_individual_risk = float(sum(individual_risks))
    diversification_benefit = (
        (total_individual_risk - volatility) / total_individual_risk
        if total_individual_risk > 0
        else 0.0
    )

    exposures_payload = {
        "weights": {inst: float(value) for inst, value in weights.items()},
        "valuations": {inst: float(value) for inst, value in context["valuations"].items()},
        "latest_prices": {
            inst: float(context["latest_prices"].get(inst, 0.0))
            for inst in weights.keys()
        },
        "methods": method_results,
        "positions": [
            {
                "instrument_id": pos.instrument_id,
                "quantity": float(pos.quantity),
                "entry_price": pos.entry_price,
            }
            for pos in context["positions"]
        ],
        "confidence": request.confidence,
        "horizon_days": request.horizon_days,
    }

    source_run = context.get("source_run")
    if source_run:
        exposures_payload["source_run"] = {
            "run_id": source_run.get("run_id"),
            "portfolio_id": source_run.get("portfolio_id"),
            "metrics": source_run.get("metrics"),
            "params": source_run.get("params"),
        }

    as_of_raw = context["as_of_date"]
    if isinstance(as_of_raw, pd.Timestamp):
        as_of_dt = as_of_raw.to_pydatetime()
    elif isinstance(as_of_raw, date):
        as_of_dt = datetime.combine(as_of_raw, datetime.min.time())
    else:
        as_of_dt = datetime.utcnow()

    now_ts = datetime.utcnow()

    var_95 = float(primary_metrics.get("var", 0.0))
    cvar_95 = float(primary_metrics.get("expected_shortfall", 0.0))
    beta = None

    if request.persist:
        try:
            var_calculator.ch_client.execute(
                """
                INSERT INTO market_intelligence.portfolio_risk_metrics
                (as_of_date, portfolio_id, run_id, volatility, variance, var_95, cvar_95, max_drawdown,
                 beta, diversification_benefit, exposures, created_at)
                VALUES
                """,
                [[
                    as_of_dt.date(),
                    context["portfolio_id"],
                    context["run_id"],
                    volatility,
                    variance,
                    var_95,
                    cvar_95,
                    max_drawdown,
                    beta,
                    diversification_benefit,
                    json.dumps(exposures_payload, default=_to_serializable),
                    now_ts,
                ]],
                types_check=True,
            )
        except Exception as exc:
            logger.exception("Failed to persist portfolio risk metrics")
            raise HTTPException(status_code=500, detail=f"Persistence failed: {exc}") from exc

    return RiskMetricsResponse(
        portfolio_id=context["portfolio_id"],
        run_id=context["run_id"],
        as_of_date=as_of_dt,
        horizon_days=request.horizon_days,
        confidence=request.confidence,
        methods=method_results,
        var_95=var_95,
        cvar_95=cvar_95,
        volatility=volatility,
        variance=variance,
        max_drawdown=max_drawdown,
        diversification_benefit=diversification_benefit,
        beta=beta,
        exposures=exposures_payload,
    )


@app.post("/api/v1/risk/stress", response_model=StressPortfolioResponse)
async def run_portfolio_stress(request: StressPortfolioRequest) -> StressPortfolioResponse:
    if not request.scenarios:
        raise HTTPException(status_code=400, detail="At least one stress scenario is required")

    context = await _build_portfolio_context(
        positions=request.positions,
        portfolio_id=request.portfolio_id,
        run_id=request.run_id,
        lookback_days=request.lookback_days,
    )

    results: List[StressScenarioResult] = []
    now_ts = datetime.utcnow()
    as_of_raw = context["as_of_date"]
    if isinstance(as_of_raw, pd.Timestamp):
        as_of_dt = as_of_raw.to_pydatetime()
    elif isinstance(as_of_raw, date):
        as_of_dt = datetime.combine(as_of_raw, datetime.min.time())
    else:
        as_of_dt = datetime.utcnow()

    inserts: List[List[Any]] = []

    for scenario in request.scenarios:
        scenario_payload = scenario.dict(by_alias=False)
        scenario_id = scenario_payload.pop("scenario_id")
        name = scenario_payload.pop("name", None)
        probability = scenario_payload.pop("probability", None)
        severity_override = scenario_payload.pop("severity", None)

        engine_payload = dict(scenario_payload)

        try:
            impact = await stress_engine.apply_scenario(context["positions"], engine_payload)
        except Exception as exc:
            logger.exception("Stress scenario %s failed", scenario_id)
            raise HTTPException(status_code=500, detail=f"Scenario {scenario_id} failed: {exc}") from exc

        pnl = float(impact.get("pnl", 0.0))
        pnl_pct = float(impact.get("pnl_pct", 0.0))
        severity = severity_override or _infer_severity(pnl_pct)
        probability_value = float(probability) if probability is not None else 0.0

        details = impact.get("details", {}) or {}
        metrics_payload = {
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "details": details,
        }

        results.append(
            StressScenarioResult(
                scenario_id=scenario_id,
                name=name,
                probability=probability_value,
                severity=severity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                details=details,
            )
        )

        if request.persist:
            inserts.append([
                as_of_dt.date(),
                context["portfolio_id"],
                context["run_id"],
                scenario_id,
                json.dumps({"name": name, **engine_payload}, default=_to_serializable),
                json.dumps(metrics_payload, default=_to_serializable),
                severity,
                probability_value,
                now_ts,
            ])

    if request.persist and inserts:
        try:
            var_calculator.ch_client.execute(
                """
                INSERT INTO market_intelligence.portfolio_stress_results
                (as_of_date, portfolio_id, run_id, scenario_id, scenario, metrics, severity, probability, created_at)
                VALUES
                """,
                inserts,
                types_check=True,
            )
        except Exception as exc:
            logger.exception("Failed to persist stress testing results")
            raise HTTPException(status_code=500, detail=f"Persistence failed: {exc}") from exc

    return StressPortfolioResponse(
        portfolio_id=context["portfolio_id"],
        run_id=context["run_id"],
        as_of_date=as_of_dt,
        results=results,
    )


@app.post("/api/v1/risk/var", response_model=VaRResponse)
async def calculate_var(request: VaRRequest):
    """
    Calculate Value at Risk for a portfolio.

    Supports multiple methods:
    - Historical: Based on historical price movements
    - Parametric: Assumes normal distribution
    - Monte Carlo: Simulated price paths
    """
    logger.info(
        f"Calculating {request.confidence_level*100}% {request.horizon_days}-day VaR "
        f"for {len(request.positions)} positions using {request.method} method"
    )
    
    try:
        # Get historical prices for all instruments
        prices_data = await var_calculator.get_historical_prices(
            [p.instrument_id for p in request.positions],
            lookback_days=252,  # 1 year
        )
        
        # Build portfolio returns
        portfolio_returns = var_calculator.build_portfolio_returns(
            request.positions,
            prices_data,
        )
        
        # Calculate VaR based on method
        if request.method == "historical":
            var_value, es_value = var_calculator.historical_var(
                portfolio_returns,
                confidence_level=request.confidence_level,
                horizon_days=request.horizon_days,
            )
        elif request.method == "parametric":
            var_value, es_value = var_calculator.parametric_var(
                portfolio_returns,
                confidence_level=request.confidence_level,
                horizon_days=request.horizon_days,
            )
        elif request.method == "monte_carlo":
            var_value, es_value = var_calculator.monte_carlo_var(
                portfolio_returns,
                confidence_level=request.confidence_level,
                horizon_days=request.horizon_days,
                n_simulations=10000,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported VaR method: {request.method}"
            )
        
        # Calculate current portfolio value
        portfolio_value = var_calculator.calculate_portfolio_value(
            request.positions,
            prices_data,
        )
        
        return VaRResponse(
            var_value=var_value,
            expected_shortfall=es_value,
            confidence_level=request.confidence_level,
            horizon_days=request.horizon_days,
            method=request.method,
            portfolio_value=portfolio_value,
            positions_count=len(request.positions),
        )
        
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/risk/stress-test", response_model=StressTestResponse)
async def run_stress_test(request: StressTestRequest):
    """
    Run stress tests on portfolio.
    
    Scenarios include:
    - Price shocks (±20%, ±50%)
    - Volatility changes
    - Correlation breakdown
    - Historical crisis events
    """
    logger.info(
        f"Running {len(request.scenarios)} stress scenarios "
        f"on {len(request.positions)} positions"
    )
    
    try:
        results = []
        
        for scenario in request.scenarios:
            impact = await stress_engine.apply_scenario(
                request.positions,
                scenario,
            )
            
            results.append({
                "scenario_name": scenario.get("name", "Unnamed"),
                "pnl": impact["pnl"],
                "pnl_pct": impact["pnl_pct"],
                "details": impact.get("details", {}),
            })
        
        # Find worst and best cases
        worst_case = min(r["pnl"] for r in results)
        best_case = max(r["pnl"] for r in results)
        
        return StressTestResponse(
            scenario_results=results,
            worst_case_loss=worst_case,
            best_case_gain=best_case,
        )
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk/correlation-matrix")
async def get_correlation_matrix(
    instrument_ids: List[str],
    start_date: date,
    end_date: date,
):
    """
    Calculate correlation matrix for instruments.
    
    Useful for understanding portfolio diversification.
    """
    try:
        prices_data = await var_calculator.get_historical_prices(
            instrument_ids,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Calculate returns
        returns_df = prices_data.pct_change().dropna()
        
        # Correlation matrix
        corr_matrix = returns_df.corr()
        
        return {
            "instruments": instrument_ids,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "correlation_matrix": corr_matrix.to_dict(),
            "average_correlation": float(
                corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).mean().mean()
            ),
        }
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk/portfolio-summary")
async def get_portfolio_summary(positions: List[Position]):
    """Get comprehensive portfolio risk summary."""
    try:
        # Calculate various risk metrics
        var_95 = await calculate_var(
            VaRRequest(
                positions=positions,
                confidence_level=0.95,
                method="historical",
            )
        )
        
        var_99 = await calculate_var(
            VaRRequest(
                positions=positions,
                confidence_level=0.99,
                method="historical",
            )
        )
        
        # Portfolio statistics
        prices_data = await var_calculator.get_historical_prices(
            [p.instrument_id for p in positions],
            lookback_days=252,
        )
        
        portfolio_returns = var_calculator.build_portfolio_returns(
            positions,
            prices_data,
        )
        
        return {
            "portfolio_value": var_95.portfolio_value,
            "positions_count": len(positions),
            "var_95_1d": var_95.var_value,
            "var_99_1d": var_99.var_value,
            "expected_shortfall_95": var_95.expected_shortfall,
            "volatility_annual": float(portfolio_returns.std() * np.sqrt(252)),
            "sharpe_ratio": float(
                portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                if portfolio_returns.std() > 0 else 0
            ),
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
