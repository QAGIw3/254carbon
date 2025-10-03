"""
LMP Decomposition Service
Separate nodal LMP into Energy, Congestion, and Loss components.
Includes PTDF calculations for congestion forecasting.
"""
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .decomposer import LMPDecomposer
from .ptdf_calculator import PTDFCalculator
from .basis_surface import BasisSurfaceModeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LMP Decomposition Service",
    description="Nodal price decomposition and congestion analysis",
    version="1.0.0",
)

# Initialize components
decomposer = LMPDecomposer()
ptdf_calc = PTDFCalculator()
basis_modeler = BasisSurfaceModeler()


class LMPComponents(BaseModel):
    """LMP decomposition components."""
    timestamp: datetime
    node_id: str
    lmp_total: float
    energy_component: float
    congestion_component: float
    loss_component: float
    currency: str = "USD"
    unit: str = "MWh"


class DecompositionRequest(BaseModel):
    """Request for LMP decomposition."""
    node_ids: List[str]
    start_time: datetime
    end_time: datetime
    iso: str  # PJM, MISO, ERCOT, etc.


class PTDFRequest(BaseModel):
    """Request for PTDF calculation."""
    source_node: str
    sink_node: str
    constraint_id: str
    iso: str


class BasisRequest(BaseModel):
    """Request for basis surface modeling."""
    hub_id: str
    node_ids: List[str]
    as_of_date: date
    iso: str


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/lmp/decompose", response_model=List[LMPComponents])
async def decompose_lmp(request: DecompositionRequest):
    """
    Decompose nodal LMP into Energy + Congestion + Loss.
    
    LMP_nodal = Energy + Congestion + Loss
    
    Where:
    - Energy: System energy price (typically hub or reference bus)
    - Congestion: Price impact of transmission constraints
    - Loss: Marginal loss component
    """
    logger.info(
        f"Decomposing LMP for {len(request.node_ids)} nodes "
        f"in {request.iso} from {request.start_time} to {request.end_time}"
    )
    
    try:
        results = []
        
        # Get raw LMP data
        lmp_data = await decomposer.get_lmp_data(
            request.node_ids,
            request.start_time,
            request.end_time,
            request.iso,
        )
        
        # Get energy component (usually from hub or reference bus)
        energy_prices = await decomposer.get_energy_component(
            request.iso,
            request.start_time,
            request.end_time,
        )
        
        # Decompose each observation
        for _, row in lmp_data.iterrows():
            node_id = row["node_id"]
            timestamp = row["timestamp"]
            lmp_total = row["lmp"]
            
            # Get energy price for this timestamp
            energy = energy_prices.get(timestamp, lmp_total * 0.95)  # Fallback: 95% of LMP
            
            # Calculate marginal loss (typically 0.5-3% of energy)
            loss = decomposer.calculate_loss_component(
                node_id,
                energy,
                request.iso,
            )
            
            # Congestion is residual
            congestion = lmp_total - energy - loss
            
            results.append(LMPComponents(
                timestamp=timestamp,
                node_id=node_id,
                lmp_total=lmp_total,
                energy_component=energy,
                congestion_component=congestion,
                loss_component=loss,
            ))
        
        logger.info(f"Decomposed {len(results)} LMP observations")
        
        return results
        
    except Exception as e:
        logger.error(f"Error decomposing LMP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/lmp/ptdf")
async def calculate_ptdf(request: PTDFRequest):
    """
    Calculate Power Transfer Distribution Factor (PTDF).
    
    PTDF measures how a 1 MW injection at source node affects
    flow on a constraint when withdrawn at sink node.
    
    Used for congestion forecasting and FTR/CRR valuation.
    """
    logger.info(
        f"Calculating PTDF from {request.source_node} to {request.sink_node} "
        f"for constraint {request.constraint_id}"
    )
    
    try:
        # Get network topology
        network = await ptdf_calc.get_network_topology(request.iso)
        
        # Calculate PTDF using DC power flow
        ptdf_value = ptdf_calc.calculate_ptdf(
            source_node=request.source_node,
            sink_node=request.sink_node,
            constraint_id=request.constraint_id,
            network=network,
        )
        
        return {
            "source_node": request.source_node,
            "sink_node": request.sink_node,
            "constraint_id": request.constraint_id,
            "ptdf_value": ptdf_value,
            "interpretation": (
                f"1 MW injection at {request.source_node} causes "
                f"{ptdf_value:.3f} MW flow on {request.constraint_id}"
            ),
        }
        
    except Exception as e:
        logger.error(f"Error calculating PTDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/lmp/basis-surface")
async def calculate_basis_surface(request: BasisRequest):
    """
    Calculate hub-to-node basis surface.
    
    Basis = Node LMP - Hub LMP
    
    Models spatial price relationships for risk management.
    """
    logger.info(
        f"Calculating basis surface from {request.hub_id} "
        f"to {len(request.node_ids)} nodes"
    )
    
    try:
        # Get historical hub and nodal prices
        hub_prices = await basis_modeler.get_hub_prices(
            request.hub_id,
            request.as_of_date,
            request.iso,
        )
        
        node_prices = await basis_modeler.get_node_prices(
            request.node_ids,
            request.as_of_date,
            request.iso,
        )
        
        # Calculate basis for each node
        basis_values = []
        
        for node_id in request.node_ids:
            basis_stats = basis_modeler.calculate_basis_statistics(
                hub_prices,
                node_prices.get(node_id, pd.Series()),
            )
            
            basis_values.append({
                "node_id": node_id,
                "mean_basis": basis_stats["mean"],
                "std_basis": basis_stats["std"],
                "percentile_95": basis_stats["p95"],
                "percentile_5": basis_stats["p5"],
                "correlation_to_hub": basis_stats["correlation"],
            })
        
        return {
            "hub_id": request.hub_id,
            "as_of_date": request.as_of_date.isoformat(),
            "basis_surface": basis_values,
        }
        
    except Exception as e:
        logger.error(f"Error calculating basis surface: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/lmp/congestion-forecast")
async def forecast_congestion(
    node_id: str,
    forecast_date: date,
    iso: str = "PJM",
):
    """
    Forecast nodal congestion component.
    
    Uses PTDF analysis and historical congestion patterns.
    """
    try:
        # Get historical congestion patterns
        historical_congestion = await decomposer.get_historical_congestion(
            node_id,
            lookback_days=90,
            iso=iso,
        )
        
        # Identify key binding constraints
        binding_constraints = await decomposer.identify_binding_constraints(
            node_id,
            iso=iso,
        )
        
        # Forecast congestion
        congestion_forecast = decomposer.forecast_nodal_congestion(
            node_id,
            forecast_date,
            historical_congestion,
            binding_constraints,
        )
        
        return {
            "node_id": node_id,
            "forecast_date": forecast_date.isoformat(),
            "iso": iso,
            "forecasted_congestion": congestion_forecast,
            "binding_constraints": binding_constraints[:5],  # Top 5
        }
        
    except Exception as e:
        logger.error(f"Error forecasting congestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)

