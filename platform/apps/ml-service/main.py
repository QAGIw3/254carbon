"""
ML Calibrator Service
Machine learning models for price forecasting and scenario calibration.
"""
import logging
import os
from datetime import date, datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# MLflow integration
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.models.signature import infer_signature

import sys
import os

import torch

# Add current directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_engineering import FeatureEngineer
from models import PriceForecastModel, ModelRegistry
from training import ModelTrainer
from retraining_pipeline import RetrainingPipeline
from research_api import router as research_router
from supply_chain_api import router as supply_chain_router
from multimodal_dataset import MultiModalDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Calibrator Service",
    description="Machine learning models for market forecasting",
    version="1.0.0",
)

# Initialize MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize components
feature_engineer = FeatureEngineer()
model_registry = ModelRegistry()
trainer = ModelTrainer()
retraining_pipeline = RetrainingPipeline()

app.include_router(research_router)
app.include_router(supply_chain_router)


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
    extras: Optional[Dict[str, Any]] = None


class TrainingRequest(BaseModel):
    instrument_ids: List[str]
    start_date: date
    end_date: date
    model_type: str = "xgboost"  # xgboost, lightgbm, transformer, ensemble
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
        
        if model.model_type == "multimodal_transformer":
            metadata = model.metadata or {}
            instrument_ids = metadata.get("instrument_ids") or [request.instrument_id]
            forecast_horizons = metadata.get("forecast_horizons") or [1, 7, 30]
            seq_len = int(metadata.get("seq_len", 64))

            feature_bundle = await feature_engineer.build_multimodal_feature_bundle(
                instrument_ids=instrument_ids,
                end=datetime.utcnow(),
                freq="1D",
            )

            dataset = MultiModalDataset(
                feature_bundle,
                seq_len=seq_len,
                forecast_horizons=forecast_horizons,
            )

            if len(dataset) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Insufficient data to generate multimodal forecast",
                )

            sample = dataset[-1]
            batch = MultiModalDataset.collate_fn([sample])

            commodity_inputs: Dict[str, Dict[str, torch.Tensor]] = {}
            commodity_masks: Dict[str, Dict[str, torch.Tensor]] = {}
            for commodity, payload in batch["commodities"].items():
                commodity_inputs[commodity] = {
                    modality: tensor.to(torch.device("cpu"))
                    for modality, tensor in payload["modalities"].items()
                }
                commodity_masks[commodity] = {
                    modality: tensor.to(torch.device("cpu"))
                    for modality, tensor in payload["masks"].items()
                }

            instrument_map = metadata.get("instrument_map") or {
                payload.get("instrument_id", commodity): commodity
                for commodity, payload in feature_bundle.items()
            }
            commodity_key = metadata.get("commodity_key") or instrument_map.get(request.instrument_id)

            if not commodity_key:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to resolve commodity for multimodal forecast",
                )

            torch_model = model.model
            torch_model.eval()

            with torch.no_grad():
                outputs = torch_model(
                    commodity_inputs,
                    commodity_masks=commodity_masks,
                    return_attentions=True,
                )

            commodity_forecasts = outputs["forecasts"].get(commodity_key)
            if commodity_forecasts is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Commodity forecasts missing for {commodity_key}",
                )

            sorted_horizons = sorted(commodity_forecasts.keys())
            forecasts = []
            confidence_intervals = []
            for horizon in sorted_horizons:
                stats = commodity_forecasts[horizon]
                mean_val = float(stats["mean"].squeeze(0).cpu().item())
                std_val = float(stats["std"].squeeze(0).cpu().item())
                interval = (
                    mean_val - 1.96 * std_val,
                    mean_val + 1.96 * std_val,
                )
                forecasts.append(
                    {
                        "month_ahead": horizon,
                        "forecast_price": mean_val,
                        "std": std_val,
                    }
                )
                confidence_intervals.append({"lower": interval[0], "upper": interval[1]})

            fusion_gates = {
                modality: float(weight.detach().cpu())
                for modality, weight in outputs["fusion_gates"].get(commodity_key, {}).items()
            }

            cross_attention = []
            if outputs.get("cross_attentions"):
                last_layer = outputs["cross_attentions"][-1]
                attn_matrix = last_layer.mean(dim=2).squeeze(2)[0].cpu().tolist()
                cross_attention = attn_matrix

            extras = {
                "fusion_gates": fusion_gates,
                "instrument_ids": instrument_ids,
                "commodity_key": commodity_key,
                "commodity_order": getattr(torch_model, "commodity_order", []),
            }
            if cross_attention:
                extras["cross_attention"] = cross_attention

            return ForecastResponse(
                instrument_id=request.instrument_id,
                forecasts=forecasts,
                model_version=model.version,
                confidence_intervals=confidence_intervals,
                extras=extras,
            )

        # Engineer features for traditional models
        features_df = await feature_engineer.build_features(
            instrument_id=request.instrument_id,
            horizon_months=request.horizon_months,
            custom_features=request.features,
        )

        predictions = model.predict(features_df)
        confidence_intervals = model.predict_interval(features_df, alpha=0.1)

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
    Train new forecasting model with MLflow tracking.

    Performs feature engineering, hyperparameter tuning, and model training.
    All training runs are tracked in MLflow for reproducibility.
    """
    logger.info(
        f"Training {request.model_type} model for "
        f"{len(request.instrument_ids)} instruments"
    )

    try:
        if request.model_type == "multimodal_transformer":
            experiment_name = f"multimodal_forecast_{'_'.join(request.instrument_ids)}"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id

            with mlflow.start_run(experiment_id=experiment_id) as run:
                mlflow.log_param("model_type", request.model_type)
                mlflow.log_param("instrument_ids", ",".join(request.instrument_ids))
                mlflow.log_param("start_date", str(request.start_date))
                mlflow.log_param("end_date", str(request.end_date))

                if request.hyperparameters:
                    for key, value in request.hyperparameters.items():
                        mlflow.log_param(f"hyperparam_{key}", value)

                multimodal_config = feature_engineer.get_multimodal_config()

                def _resolve_head_mask_config() -> Dict[str, List[int]]:
                    head_masks: Dict[str, List[int]] = {}
                    defaults = multimodal_config.get("defaults", {}).get("commodity_groups", {})
                    for commodity, payload in feature_bundle.items():
                        group = payload.get("commodity_group")
                        if not group:
                            continue
                        group_cfg = defaults.get(group, {})
                        fused_heads = group_cfg.get("fused_heads")
                        if fused_heads:
                            indices = [int(h) for h in fused_heads]
                            head_masks.setdefault(group, indices)
                            head_masks.setdefault(commodity, indices)
                    return head_masks

                feature_bundle = await feature_engineer.build_multimodal_feature_bundle(
                    instrument_ids=request.instrument_ids,
                    start=datetime.combine(request.start_date, datetime.min.time()),
                    end=datetime.combine(request.end_date, datetime.min.time()),
                    freq="1D",
                )

                head_mask_config = _resolve_head_mask_config()

                model, metrics, artifacts = await trainer.train_multimodal_transformer(
                    feature_bundle=feature_bundle,
                    hyperparameters=request.hyperparameters,
                    head_mask_config=head_mask_config,
                )

                mlflow.log_param("seq_len", artifacts["seq_len"])
                mlflow.log_param(
                    "forecast_horizons",
                    ",".join(str(h) for h in artifacts["forecast_horizons"]),
                )
                for key, value in artifacts["hyperparameters"].items():
                    mlflow.log_param(f"hyper_{key}", value)

                mlflow.log_metric("val_loss", metrics["val_loss"])
                aggregate_metrics = metrics.get("aggregate", {})
                if aggregate_metrics:
                    mlflow.log_metric("aggregate_mape", aggregate_metrics.get("mape", 0.0))
                    mlflow.log_metric("aggregate_rmse", aggregate_metrics.get("rmse", 0.0))
                for commodity, commodity_metrics in metrics.get("per_commodity", {}).items():
                    if "mape" in commodity_metrics:
                        mlflow.log_metric(f"{commodity}_mape", commodity_metrics["mape"])
                    if "rmse" in commodity_metrics:
                        mlflow.log_metric(f"{commodity}_rmse", commodity_metrics["rmse"])

                mlflow.log_dict(artifacts.get("fusion_snapshot", {}), "fusion_snapshot.json")
                mlflow.pytorch.log_model(model, "multimodal_transformer_model")

                results = []
                for instrument_id in request.instrument_ids:
                    commodity_key = artifacts["instrument_map"].get(instrument_id)
                    commodity_metrics = metrics.get("per_commodity", {}).get(commodity_key, {})
                    registry_metrics = {
                        "val_loss": float(metrics["val_loss"]),
                    }
                    if "mape" in commodity_metrics:
                        registry_metrics["mape"] = float(commodity_metrics["mape"])
                    if "rmse" in commodity_metrics:
                        registry_metrics["rmse"] = float(commodity_metrics["rmse"])

                    metadata = {
                        "instrument_ids": request.instrument_ids,
                        "commodity_key": commodity_key,
                        "instrument_map": artifacts["instrument_map"],
                        "commodity_groups": artifacts["commodity_groups"],
                        "seq_len": artifacts["seq_len"],
                        "forecast_horizons": artifacts["forecast_horizons"],
                        "hyperparameters": artifacts["hyperparameters"],
                        "fusion_snapshot": artifacts.get("fusion_snapshot", {}),
                        "head_mask_config": head_mask_config,
                    }

                    model_version = model_registry.register(
                        instrument_id=instrument_id,
                        model=model,
                        metrics=registry_metrics,
                        model_type=request.model_type,
                        metadata=metadata,
                    )

                    results.append({
                        "instrument_id": instrument_id,
                        "model_version": model_version,
                        "mlflow_run_id": run.info.run_id,
                        "metrics": registry_metrics,
                        "commodity": commodity_key,
                    })

                return {
                    "status": "success",
                    "models_trained": len(results),
                    "results": results,
                }

        results = []

        for instrument_id in request.instrument_ids:
            # Create MLflow experiment for this instrument
            experiment_name = f"price_forecast_{instrument_id}"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id

            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log training parameters
                mlflow.log_param("instrument_id", instrument_id)
                mlflow.log_param("model_type", request.model_type)
                mlflow.log_param("start_date", str(request.start_date))
                mlflow.log_param("end_date", str(request.end_date))

                if request.hyperparameters:
                    for key, value in request.hyperparameters.items():
                        mlflow.log_param(f"hyperparam_{key}", value)

                # Fetch historical data and features
                training_data = await feature_engineer.build_training_dataset(
                    instrument_id=instrument_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )

                # Log dataset info
                mlflow.log_param("training_samples", len(training_data))
                mlflow.log_param(
                    "feature_count",
                    training_data.shape[1]
                    if hasattr(training_data, "shape")
                    else len(training_data.columns),
                )

                # Train model with hyperparameter tuning
                model, metrics = await trainer.train(
                    instrument_id=instrument_id,
                    data=training_data,
                    model_type=request.model_type,
                    hyperparameters=request.hyperparameters,
                )

                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log the model
                feature_cols = training_data.drop('target', axis=1) if 'target' in training_data else training_data.drop('target_price', axis=1, errors='ignore')
                signature = infer_signature(feature_cols, model.predict(feature_cols))
                if hasattr(model, 'model'):
                    mlflow.sklearn.log_model(
                        model.model,
                        f"model_{instrument_id}",
                        signature=signature
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        f"model_{instrument_id}",
                        signature=signature
                    )

                # Register model in local registry
                model_version = model_registry.register(
                    instrument_id=instrument_id,
                    model=model,
                    metrics=metrics,
                )

                # Log model registry info
                mlflow.log_param("registry_version", model_version)

                results.append({
                    "instrument_id": instrument_id,
                    "model_version": model_version,
                    "mlflow_run_id": run.info.run_id,
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
