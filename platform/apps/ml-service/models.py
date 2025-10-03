"""
ML model implementations and registry.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pickle
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


class PriceForecastModel:
    """Wrapper for price forecasting models."""
    
    def __init__(
        self,
        model: Any,
        version: str,
        model_type: str,
        instrument_id: str,
        metrics: Dict[str, float],
    ):
        self.model = model
        self.version = version
        self.model_type = model_type
        self.instrument_id = instrument_id
        self.metrics = metrics
        self.trained_date = datetime.utcnow()
        self.is_active = False
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""
        # Drop non-feature columns
        feature_cols = [
            c for c in features.columns
            if c not in ["date", "forecast_date", "month_ahead", "target_price"]
        ]
        
        X = features[feature_cols]
        return self.model.predict(X)
    
    def predict_interval(
        self,
        features: pd.DataFrame,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.
        
        Uses quantile regression or bootstrap for uncertainty estimation.
        """
        predictions = self.predict(features)
        
        # Simple approach: use residual standard error
        # In production, would use proper quantile regression
        std_error = np.sqrt(self.metrics.get("mse", 0.0))
        z_score = 1.645  # 90% confidence interval
        
        margin = z_score * std_error
        
        ci_lower = predictions - margin
        ci_upper = predictions + margin
        
        return ci_lower, ci_upper
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(path, exist_ok=True)
        
        model_path = os.path.join(path, f"{self.instrument_id}_{self.version}.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "version": self.version,
                    "model_type": self.model_type,
                    "instrument_id": self.instrument_id,
                    "metrics": self.metrics,
                    "trained_date": self.trained_date,
                },
                f,
            )
        
        logger.info(f"Model saved: {model_path}")
    
    @classmethod
    def load(cls, path: str) -> "PriceForecastModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        return cls(
            model=data["model"],
            version=data["version"],
            model_type=data["model_type"],
            instrument_id=data["instrument_id"],
            metrics=data["metrics"],
        )


class ModelRegistry:
    """
    Model registry for versioning and deployment.
    
    Manages multiple model versions per instrument.
    """
    
    def __init__(self, storage_path: str = "/models"):
        self.storage_path = storage_path
        self.models: Dict[str, Dict[str, PriceForecastModel]] = {}
        self._load_models()
    
    def _load_models(self):
        """Load all saved models from disk."""
        if not os.path.exists(self.storage_path):
            logger.info("No saved models found")
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".pkl"):
                try:
                    model_path = os.path.join(self.storage_path, filename)
                    model = PriceForecastModel.load(model_path)
                    
                    if model.instrument_id not in self.models:
                        self.models[model.instrument_id] = {}
                    
                    self.models[model.instrument_id][model.version] = model
                    logger.info(f"Loaded model: {filename}")
                except Exception as e:
                    logger.error(f"Error loading model {filename}: {e}")
    
    def register(
        self,
        instrument_id: str,
        model: Any,
        metrics: Dict[str, float],
        model_type: str = "xgboost",
    ) -> str:
        """Register new model version."""
        # Generate version
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        wrapped_model = PriceForecastModel(
            model=model,
            version=version,
            model_type=model_type,
            instrument_id=instrument_id,
            metrics=metrics,
        )
        
        # Save to disk
        wrapped_model.save(self.storage_path)
        
        # Add to registry
        if instrument_id not in self.models:
            self.models[instrument_id] = {}
        
        self.models[instrument_id][version] = wrapped_model
        
        # Auto-activate if first model
        if len(self.models[instrument_id]) == 1:
            wrapped_model.is_active = True
        
        logger.info(
            f"Registered model {version} for {instrument_id}: "
            f"MAPE={metrics.get('mape', 0):.2f}%"
        )
        
        return version
    
    def get_model(
        self,
        instrument_id: str,
        version: Optional[str] = None,
    ) -> Optional[PriceForecastModel]:
        """Get model by instrument and version (defaults to active)."""
        if instrument_id not in self.models:
            return None
        
        if version:
            return self.models[instrument_id].get(version)
        
        # Return active model
        for model in self.models[instrument_id].values():
            if model.is_active:
                return model
        
        # Return latest if no active
        versions = sorted(self.models[instrument_id].keys(), reverse=True)
        if versions:
            return self.models[instrument_id][versions[0]]
        
        return None
    
    def activate(self, instrument_id: str, version: str):
        """Set model version as active."""
        if instrument_id not in self.models:
            raise ValueError(f"No models for {instrument_id}")
        
        if version not in self.models[instrument_id]:
            raise ValueError(f"Version {version} not found")
        
        # Deactivate all
        for model in self.models[instrument_id].values():
            model.is_active = False
        
        # Activate specified version
        self.models[instrument_id][version].is_active = True
        
        logger.info(f"Activated model {version} for {instrument_id}")
    
    def list_models(self, instrument_id: str) -> List[PriceForecastModel]:
        """List all model versions for an instrument."""
        if instrument_id not in self.models:
            return []
        
        return list(self.models[instrument_id].values())
    
    def count(self) -> int:
        """Total number of models loaded."""
        return sum(len(models) for models in self.models.values())

