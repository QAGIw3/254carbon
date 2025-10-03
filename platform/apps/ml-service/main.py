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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)

