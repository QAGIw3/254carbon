"""
ML Calibrator Service
Machine learning models for price forecasting and scenario calibration.
"""
import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .feature_engineering import FeatureEngineer
from .models import PriceForecastModel, ModelRegistry
from .training import ModelTrainer
from .retraining_pipeline import RetrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Calibrator Service",
    description="Machine learning models for market forecasting",
    version="1.0.0",
)

# Initialize components
feature_engineer = FeatureEngineer()
model_registry = ModelRegistry()
trainer = ModelTrainer()
retraining_pipeline = RetrainingPipeline()


class ForecastRequest(BaseModel):
    instrument_id: str
    horizon_months: int = 12
    features: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = None


class ForecastResponse(BaseModel):
    instrument_id: str
    forecasts: List[Dict[str, Any]]
    model_version: str
    confidence_intervals: List[Dict[str, float]]


class TrainingRequest(BaseModel):
    instrument_ids: List[str]
    start_date: date
    end_date: date
    model_type: str = "xgboost"  # xgboost, lightgbm, ensemble
    hyperparameters: Optional[Dict[str, Any]] = None


class ModelMetrics(BaseModel):
    model_version: str
    instrument_id: str
    mape: float
    rmse: float
    r2_score: float
    training_date: datetime


@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": model_registry.count()}


@app.post("/api/v1/ml/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate ML-based price forecast.
    
    Uses trained models with fundamental features to predict future prices.
    """
    logger.info(
        f"Generating forecast for {request.instrument_id}, "
        f"horizon={request.horizon_months} months"
    )
    
    try:
        # Get or load model
        model = model_registry.get_model(
            request.instrument_id,
            version=request.model_version,
        )
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for {request.instrument_id}"
            )
        
        # Engineer features
        features_df = await feature_engineer.build_features(
            instrument_id=request.instrument_id,
            horizon_months=request.horizon_months,
            custom_features=request.features,
        )
        
        # Generate predictions
        predictions = model.predict(features_df)
        confidence_intervals = model.predict_interval(features_df, alpha=0.1)
        
        # Format response
        forecasts = []
        for i, (pred, ci_low, ci_high) in enumerate(
            zip(predictions, confidence_intervals[0], confidence_intervals[1])
        ):
            forecasts.append({
                "month_ahead": i + 1,
                "forecast_price": float(pred),
                "ci_lower": float(ci_low),
                "ci_upper": float(ci_high),
            })
        
        return ForecastResponse(
            instrument_id=request.instrument_id,
            forecasts=forecasts,
            model_version=model.version,
            confidence_intervals=[
                {"lower": float(ci[0]), "upper": float(ci[1])}
                for ci in zip(confidence_intervals[0], confidence_intervals[1])
            ],
        )
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/train")
async def train_model(request: TrainingRequest):
    """
    Train new forecasting model.
    
    Performs feature engineering, hyperparameter tuning, and model training.
    """
    logger.info(
        f"Training {request.model_type} model for "
        f"{len(request.instrument_ids)} instruments"
    )
    
    try:
        results = []
        
        for instrument_id in request.instrument_ids:
            # Fetch historical data and features
            training_data = await feature_engineer.build_training_dataset(
                instrument_id=instrument_id,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            
            # Train model with hyperparameter tuning
            model, metrics = await trainer.train(
                instrument_id=instrument_id,
                data=training_data,
                model_type=request.model_type,
                hyperparameters=request.hyperparameters,
            )
            
            # Register model
            model_version = model_registry.register(
                instrument_id=instrument_id,
                model=model,
                metrics=metrics,
            )
            
            results.append({
                "instrument_id": instrument_id,
                "model_version": model_version,
                "metrics": metrics,
            })
        
        return {
            "status": "success",
            "models_trained": len(results),
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/models/{instrument_id}")
async def get_model_info(instrument_id: str):
    """Get information about trained models for an instrument."""
    models = model_registry.list_models(instrument_id)
    
    return {
        "instrument_id": instrument_id,
        "models": [
            {
                "version": m.version,
                "model_type": m.model_type,
                "trained_date": m.trained_date,
                "metrics": m.metrics,
                "is_active": m.is_active,
            }
            for m in models
        ],
    }


@app.post("/api/v1/ml/models/{instrument_id}/activate")
async def activate_model(instrument_id: str, version: str):
    """Activate a specific model version for production use."""
    try:
        model_registry.activate(instrument_id, version)
        return {
            "status": "success",
            "instrument_id": instrument_id,
            "active_version": version,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/models/retrain")
async def trigger_retraining(
    instrument_ids: List[str],
    retrain_threshold: float = 0.15,  # Retrain if MAPE > 15%
    force_retrain: bool = False,
):
    """
    Trigger automated model retraining pipeline.

    Monitors model performance and retrains when performance degrades.
    """
    logger.info(f"Triggering retraining for {len(instrument_ids)} instruments")

    results = []

    for instrument_id in instrument_ids:
        # Check current model performance
        current_performance = await retraining_pipeline.check_model_performance(
            instrument_id
        )

        should_retrain = (
            force_retrain or
            current_performance.get("mape", 0) > retrain_threshold
        )

        if should_retrain:
            logger.info(f"Retraining {instrument_id} (MAPE: {current_performance.get('mape', 0):.3f})")

            # Run retraining pipeline
            retrain_result = await retraining_pipeline.retrain_model(
                instrument_id,
                model_type="xgboost"  # Default model type
            )

            results.append({
                "instrument_id": instrument_id,
                "retrained": True,
                "old_performance": current_performance,
                "new_performance": retrain_result.get("performance"),
                "model_version": retrain_result.get("model_version"),
            })
        else:
            logger.info(f"Skipping {instrument_id} (MAPE: {current_performance.get('mape', 0):.3f})")

            results.append({
                "instrument_id": instrument_id,
                "retrained": False,
                "current_performance": current_performance,
            })

    return {
        "total_instruments": len(instrument_ids),
        "retrained_count": sum(1 for r in results if r["retrained"]),
        "results": results,
    }


@app.get("/api/v1/models/performance")
async def get_model_performance(
    instrument_id: str,
    days: int = 30,
):
    """
    Get model performance metrics for monitoring.

    Returns MAPE, RMSE, and other metrics over recent period.
    """
    try:
        performance = await retraining_pipeline.get_performance_metrics(
            instrument_id, days=days
        )

        return {
            "instrument_id": instrument_id,
            "evaluation_period_days": days,
            "metrics": performance,
        }

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/health")
async def get_models_health():
    """
    Get overall health status of all deployed models.

    Returns summary of model performance across all instruments.
    """
    try:
        # Get list of all instruments from registry
        all_instruments = await model_registry.get_all_instruments()

        health_summary = {
            "total_models": len(all_instruments),
            "healthy_models": 0,
            "degraded_models": 0,
            "failed_models": 0,
            "instrument_details": [],
        }

        for instrument_id in all_instruments:
            try:
                performance = await retraining_pipeline.check_model_performance(instrument_id)

                mape = performance.get("mape", 0)
                instrument_health = {
                    "instrument_id": instrument_id,
                    "mape": mape,
                    "rmse": performance.get("rmse", 0),
                    "status": "healthy" if mape < 0.12 else "degraded" if mape < 0.20 else "failed",
                }

                health_summary["instrument_details"].append(instrument_health)

                if mape < 0.12:
                    health_summary["healthy_models"] += 1
                elif mape < 0.20:
                    health_summary["degraded_models"] += 1
                else:
                    health_summary["failed_models"] += 1

            except Exception as e:
                logger.warning(f"Could not check health for {instrument_id}: {e}")
                health_summary["instrument_details"].append({
                    "instrument_id": instrument_id,
                    "status": "unknown",
                    "error": str(e),
                })

        return health_summary

    except Exception as e:
        logger.error(f"Error getting models health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)

