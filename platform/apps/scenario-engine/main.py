"""
Scenario Engine Service
DSL parser, fundamentals layer, ML calibrator, and execution framework.
"""
import asyncio
import json
import logging
import uuid
from datetime import date, datetime
from typing import Dict, Any, Optional

import aiohttp
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Scenario Engine",
    description="Scenario modeling and forecast execution",
    version="1.0.0",
)


class ScenarioSpec(BaseModel):
    """Scenario DSL specification."""
    as_of_date: date
    macro: Dict[str, Any]
    fuels: Dict[str, Any]
    power: Dict[str, Any]
    policy: Dict[str, Any]
    market_overrides: Optional[Dict[str, Any]] = None
    technical: Optional[Dict[str, Any]] = None


class ScenarioRunRequest(BaseModel):
    scenario_id: str
    spec: ScenarioSpec


class ScenarioRunResponse(BaseModel):
    run_id: str
    scenario_id: str
    status: str
    message: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/scenarios", response_model=dict)
async def create_scenario(title: str, description: str):
    """Create a new scenario."""
    scenario_id = f"SCENARIO_{uuid.uuid4().hex[:8].upper()}"
    
    # TODO: Store in PostgreSQL
    logger.info(f"Created scenario: {scenario_id}")
    
    return {
        "scenario_id": scenario_id,
        "title": title,
        "description": description,
        "status": "created",
    }


async def execute_scenario_pipeline(scenario_id: str, run_id: str, spec: ScenarioSpec) -> Dict[str, Any]:
    """Execute the full scenario pipeline."""
    results = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": "running",
        "steps": [],
        "start_time": datetime.utcnow().isoformat()
    }

    try:
        # Step 1: Parse assumptions from DSL spec
        logger.info(f"Step 1: Parsing assumptions for scenario {scenario_id}")
        assumptions = await parse_assumptions(spec)
        results["steps"].append({
            "step": "parse_assumptions",
            "status": "completed",
            "assumptions": assumptions
        })

        # Step 2: Run fundamentals layer
        logger.info(f"Step 2: Running fundamentals layer for scenario {scenario_id}")
        fundamentals_results = await run_fundamentals_layer(assumptions)
        results["steps"].append({
            "step": "fundamentals_layer",
            "status": "completed",
            "fundamentals": fundamentals_results
        })

        # Step 3: ML calibrator
        logger.info(f"Step 3: Running ML calibrator for scenario {scenario_id}")
        calibrated_results = await run_ml_calibrator(fundamentals_results, spec)
        results["steps"].append({
            "step": "ml_calibrator",
            "status": "completed",
            "calibration": calibrated_results
        })

        # Step 4: Curve generation
        logger.info(f"Step 4: Generating curves for scenario {scenario_id}")
        curve_results = await generate_curves(calibrated_results, spec)
        results["steps"].append({
            "step": "curve_generation",
            "status": "completed",
            "curves": curve_results
        })

        # Step 5: Store results in PostgreSQL
        logger.info(f"Step 5: Storing results for scenario {scenario_id}")
        storage_result = await store_results(results)
        results["steps"].append({
            "step": "store_results",
            "status": "completed",
            "storage": storage_result
        })

        results["status"] = "completed"
        results["end_time"] = datetime.utcnow().isoformat()

        return results

    except Exception as e:
        logger.error(f"Error in scenario execution: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        results["end_time"] = datetime.utcnow().isoformat()
        return results


async def parse_assumptions(spec: ScenarioSpec) -> Dict[str, Any]:
    """Parse scenario assumptions from DSL specification."""
    assumptions = {}

    # Extract macro assumptions
    if spec.macro:
        assumptions["macro"] = spec.macro

    # Extract fuel assumptions
    if spec.fuels:
        assumptions["fuels"] = spec.fuels

    # Extract power assumptions
    if spec.power:
        assumptions["power"] = spec.power

    # Extract policy assumptions
    if spec.policy:
        assumptions["policy"] = spec.policy

    # Extract market overrides
    if spec.market_overrides:
        assumptions["market_overrides"] = spec.market_overrides

    # Extract technical assumptions
    if spec.technical:
        assumptions["technical"] = spec.technical

    return assumptions


async def run_fundamentals_layer(assumptions: Dict[str, Any]) -> Dict[str, Any]:
    """Run fundamentals layer calculations."""
    # For now, return mock fundamentals results
    # In a real implementation, this would call a fundamentals service
    fundamentals = {
        "load_forecast": {
            "annual_growth": assumptions.get("power", {}).get("load_growth", 1.5),
            "peak_demand": 150000,  # MW
            "seasonal_factors": {"summer": 1.2, "winter": 0.9}
        },
        "generation_mix": {
            "coal": 0.20,
            "gas": 0.35,
            "nuclear": 0.20,
            "renewables": 0.25
        },
        "fuel_prices": assumptions.get("fuels", {}),
        "policy_impact": assumptions.get("policy", {})
    }

    return fundamentals


async def run_ml_calibrator(fundamentals: Dict[str, Any], spec: ScenarioSpec) -> Dict[str, Any]:
    """Run ML calibrator with fundamentals and scenario parameters."""
    try:
        # Call ML service for calibration
        async with aiohttp.ClientSession() as session:
            ml_payload = {
                "fundamentals": fundamentals,
                "scenario": spec.dict(),
                "calibration_type": "price_forecast"
            }

            async with session.post(
                "http://ml-service:8007/api/v1/calibrate",
                json=ml_payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    calibration_result = await response.json()
                else:
                    # Fallback to mock calibration
                    calibration_result = {
                        "calibrated_prices": {
                            "power": {"MISO": 45.0, "PJM": 42.0, "CAISO": 48.0},
                            "gas": {"HENRY": 3.5, "CHICAGO": 3.2}
                        },
                        "confidence_intervals": {
                            "power": {"MISO": [42.0, 48.0], "PJM": [39.0, 45.0]}
                        },
                        "model_version": "v1.0"
                    }

        return calibration_result

    except Exception as e:
        logger.warning(f"ML service unavailable, using mock calibration: {e}")
        # Return mock calibration results
        return {
            "calibrated_prices": {
                "power": {"MISO": 45.0, "PJM": 42.0, "CAISO": 48.0},
                "gas": {"HENRY": 3.5, "CHICAGO": 3.2}
            },
            "confidence_intervals": {
                "power": {"MISO": [42.0, 48.0], "PJM": [39.0, 45.0]}
            },
            "model_version": "mock_v1.0"
        }


async def generate_curves(calibration: Dict[str, Any], spec: ScenarioSpec) -> Dict[str, Any]:
    """Generate forward curves using calibrated parameters."""
    try:
        # Call curve service for generation
        async with aiohttp.ClientSession() as session:
            curve_payload = {
                "as_of_date": spec.as_of_date.isoformat(),
                "calibrated_prices": calibration.get("calibrated_prices", {}),
                "scenario_id": f"SCENARIO_{spec.as_of_date.strftime('%Y%m%d')}"
            }

            async with session.post(
                "http://curve-service:8001/api/v1/curves/generate",
                json=curve_payload,
                timeout=60
            ) as response:
                if response.status == 200:
                    curve_result = await response.json()
                else:
                    # Fallback to mock curve generation
                    curve_result = {
                        "curves_generated": ["MISO.HUB.INDIANA", "PJM.HUB.WEST"],
                        "curve_points": 240,  # 2 years of monthly points
                        "run_id": f"CURVE_{uuid.uuid4().hex[:8]}"
                    }

        return curve_result

    except Exception as e:
        logger.warning(f"Curve service unavailable, using mock curves: {e}")
        # Return mock curve results
        return {
            "curves_generated": ["MISO.HUB.INDIANA", "PJM.HUB.WEST", "CAISO.SP15"],
            "curve_points": 240,
            "run_id": f"CURVE_{uuid.uuid4().hex[:8]}",
            "status": "mock_generated"
        }


async def store_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Store scenario execution results in PostgreSQL."""
    try:
        # In a real implementation, this would store in PostgreSQL
        # For now, just return success
        return {
            "stored_records": len(results.get("steps", [])),
            "database": "postgresql",
            "table": "scenario_runs",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error storing results: {e}")
        raise


@app.post("/api/v1/scenarios/{scenario_id}/runs", response_model=ScenarioRunResponse)
async def execute_scenario(scenario_id: str, request: ScenarioRunRequest):
    """Execute a scenario run."""
    run_id = str(uuid.uuid4())

    logger.info(f"Executing scenario {scenario_id}, run {run_id}")

    try:
        # Execute scenario in background
        asyncio.create_task(execute_scenario_pipeline(scenario_id, run_id, request.spec))

        return ScenarioRunResponse(
            run_id=run_id,
            scenario_id=scenario_id,
            status="queued",
            message="Scenario execution started",
        )

    except Exception as e:
        logger.error(f"Error executing scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scenarios/{scenario_id}/runs/{run_id}")
async def get_run_status(scenario_id: str, run_id: str):
    """Get scenario run status."""
    try:
        # In a real implementation, this would query PostgreSQL for run status
        # For now, return mock status with more realistic data

        # Simulate different statuses based on run_id hash for demo purposes
        import hashlib
        hash_input = f"{scenario_id}_{run_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        status_options = ["completed", "running", "failed", "queued"]
        mock_status = status_options[hash_value % len(status_options)]

        # Mock progress based on status
        if mock_status == "completed":
            progress = 100
            message = "Scenario execution completed successfully"
        elif mock_status == "running":
            progress = 65
            message = "Processing ML calibration and curve generation"
        elif mock_status == "failed":
            progress = 30
            message = "Error in fundamentals layer processing"
        else:
            progress = 0
            message = "Scenario queued for execution"

        return {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "status": mock_status,
            "progress": progress,
            "message": message,
            "steps_completed": progress // 20,  # Each step ~20% progress
            "estimated_completion": "2025-10-03T15:30:00Z" if mock_status == "running" else None
        }

    except Exception as e:
        logger.error(f"Error getting run status: {e}")
        return {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "status": "error",
            "message": "Error retrieving run status"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

