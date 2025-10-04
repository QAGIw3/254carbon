"""
Curve Service
Forward curve generation with QP smoothing, tenor reconciliation, and lineage tracking.
"""
import logging
import uuid
from datetime import date, datetime
from typing import Optional

import numpy as np
import cvxpy as cp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
from qp_solver import smooth_curve, reconcile_tenors
import os
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Curve Service",
    description="Forward curve generation and smoothing",
    version="1.0.0",
)


class CurveRequest(BaseModel):
    instrument_id: str
    as_of_date: date
    scenario_id: str = "BASE"
    targets: list[float]  # μ_t fundamentals targets
    bounds_lower: Optional[list[float]] = None
    bounds_upper: Optional[list[float]] = None
    smoothness_lambda: float = 50.0


class CurveResponse(BaseModel):
    run_id: str
    instrument_id: str
    scenario_id: str
    points: list[dict]
    status: str
    message: Optional[str] = None
    decomposition_sample: Optional[dict] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@app.post("/api/v1/curves/generate", response_model=CurveResponse)
async def generate_curve(request: CurveRequest):
    """
    Generate forward curve using QP smoothing.
    
    Minimizes: Σ_t w_t (p_t - μ_t)^2 + λ Σ_t (Δ² p_t)^2
    Subject to: bid_t ≤ p_t ≤ ask_t, p_t ≥ 0
    """
    logger.info(
        f"Generating curve for {request.instrument_id}, "
        f"scenario {request.scenario_id}"
    )
    
    run_id = str(uuid.uuid4())
    
    try:
        # Get instrument metadata
        pool = await db.get_postgres_pool()
        async with pool.acquire() as conn:
            instrument = await conn.fetchrow(
                "SELECT * FROM pg.instrument WHERE instrument_id = $1",
                request.instrument_id,
            )
            
            if not instrument:
                raise HTTPException(
                    status_code=404,
                    detail=f"Instrument not found: {request.instrument_id}",
                )
        
        # Prepare bounds
        n_points = len(request.targets)
        lb = (
            np.array(request.bounds_lower)
            if request.bounds_lower
            else np.zeros(n_points)
        )
        ub = (
            np.array(request.bounds_upper)
            if request.bounds_upper
            else np.full(n_points, np.inf)
        )
        
        # Solve QP
        prices = smooth_curve(
            mu=np.array(request.targets),
            lb=lb,
            ub=ub,
            lam=request.smoothness_lambda,
        )
        
        if prices is None:
            # Try relaxing bounds
            logger.warning("Initial solve failed, relaxing bounds by 5%")
            lb = lb * 0.95
            ub = ub * 1.05
            prices = smooth_curve(
                mu=np.array(request.targets),
                lb=lb,
                ub=ub,
                lam=request.smoothness_lambda,
            )
            
            if prices is None:
                raise HTTPException(
                    status_code=500,
                    detail="Curve generation failed (infeasible)",
                )
        
        # Reconcile tenors (monthly -> quarterly -> annual)
        reconciled = reconcile_tenors(prices)
        
        # Store in ClickHouse
        ch_client = db.get_clickhouse_client()
        
        points = []
        for i, price in enumerate(reconciled):
            # Generate delivery dates (simplified - would use actual tenor logic)
            delivery_start = request.as_of_date
            delivery_end = request.as_of_date
            
            point = {
                "as_of_date": request.as_of_date,
                "market": instrument["market"],
                "product": instrument["product"],
                "instrument_id": request.instrument_id,
                "curve_id": f"{request.instrument_id}_FORWARD",
                "scenario_id": request.scenario_id,
                "delivery_start": delivery_start,
                "delivery_end": delivery_end,
                "tenor_type": "Month",  # Simplified
                "price": float(price),
                "currency": instrument["currency"],
                "unit": instrument["unit"],
                "source": "curve_service",
                "run_id": run_id,
                "version_id": 1,
            }
            points.append(point)
        
        # Bulk insert
        if points:
            ch_client.execute(
                "INSERT INTO ch.forward_curve_points VALUES",
                points,
            )
        
        logger.info(f"Curve generated successfully: {run_id}")

        # Optionally enrich with LMP decomposition sample (first node if available)
        decomposition_sample = None
        try:
            if os.getenv("ENABLE_DECOMPOSITION", "0") == "1":
                # Derive a plausible node_id from instrument (if already node-like)
                node_id = request.instrument_id
                start_time = datetime.combine(request.as_of_date, datetime.min.time())
                end_time = datetime.combine(request.as_of_date, datetime.max.time())
                payload = {
                    "node_ids": [node_id],
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "iso": instrument["market"].upper(),
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        "http://lmp-decomposition-service:8009/api/v1/lmp/decompose",
                        json=payload,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, list) and data:
                            sample = data[0]
                            decomposition_sample = {
                                "timestamp": sample.get("timestamp"),
                                "node_id": sample.get("node_id"),
                                "lmp_total": sample.get("lmp_total"),
                                "energy_component": sample.get("energy_component"),
                                "congestion_component": sample.get("congestion_component"),
                                "loss_component": sample.get("loss_component"),
                            }
        except Exception as e:
            logger.warning(f"Decomposition enrichment skipped: {e}")
        
        return CurveResponse(
            run_id=run_id,
            instrument_id=request.instrument_id,
            scenario_id=request.scenario_id,
            points=points,
            status="success",
            decomposition_sample=decomposition_sample,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/curves/status/{run_id}")
async def get_curve_status(run_id: str):
    """Get status of a curve generation run."""
    try:
        ch_client = db.get_clickhouse_client()
        
        result = ch_client.execute(
            """
            SELECT COUNT(*) as count
            FROM ch.forward_curve_points
            WHERE run_id = %(run_id)s
            """,
            {"run_id": run_id},
        )
        
        count = result[0][0] if result else 0
        
        return {
            "run_id": run_id,
            "status": "completed" if count > 0 else "not_found",
            "points_count": count,
        }
    except Exception as e:
        logger.error(f"Error fetching curve status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

