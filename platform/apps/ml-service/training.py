"""
Model training with hyperparameter tuning.
"""
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and tune forecasting models."""
    
    def __init__(self):
        self.default_params = {
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
            "lightgbm": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 63, 127],
            },
        }
    
    async def train(
        self,
        instrument_id: str,
        data: pd.DataFrame,
        model_type: str = "xgboost",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train model with hyperparameter tuning.
        
        Uses time-series cross-validation to avoid lookahead bias.
        """
        logger.info(f"Training {model_type} model for {instrument_id}")
        
        # Prepare features and target
        X, y = self._prepare_data(data)
        
        # Get base model
        base_model = self._get_base_model(model_type)
        
        # Hyperparameter tuning
        if hyperparameters is None:
            hyperparameters = self.default_params.get(model_type, {})
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            base_model,
            hyperparameters,
            cv=tscv,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=-1,
            verbose=1,
        )
        
        grid_search.fit(X, y)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Calculate metrics on full dataset
        y_pred = best_model.predict(X)
        
        metrics = {
            "mape": float(mean_absolute_percentage_error(y, y_pred) * 100),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mse": float(mean_squared_error(y, y_pred)),
            "r2_score": float(r2_score(y, y_pred)),
            "best_params": grid_search.best_params_,
        }
        
        logger.info(
            f"Model trained successfully: "
            f"MAPE={metrics['mape']:.2f}%, "
            f"RMSE={metrics['rmse']:.2f}, "
            f"R²={metrics['r2_score']:.3f}"
        )
        
        return best_model, metrics
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from training data."""
        # Drop non-feature columns
        exclude_cols = ["date", "forecast_date", "target_price"]
        
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        X = data[feature_cols]
        y = data["target_price"]
        
        return X, y
    
    def _get_base_model(self, model_type: str):
        """Get base model instance."""
        if model_type == "xgboost":
            return XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "lightgbm":
            return LGBMRegressor(
                objective="regression",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

