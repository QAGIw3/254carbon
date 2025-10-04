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

# Transformer model imports
from models import TransformerPriceForecastModel

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
            "transformer": {
                "hidden_size": [128, 256, 512],
                "num_layers": [3, 6, 9],
                "num_heads": [4, 8, 16],
                "dropout": [0.1, 0.2],
                "learning_rate": [1e-4, 5e-4, 1e-3],
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

        if model_type == "transformer":
            # Train transformer model separately
            best_model, metrics = await self._train_transformer_model(
                instrument_id, X, y, hyperparameters
            )
        else:
            # Get base model for traditional ML models
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
        
        if model_type == "transformer":
            # Metrics already calculated in _train_transformer_model
            pass
        else:
            # Calculate metrics on full dataset for traditional models
            y_pred = best_model.predict(X)

            metrics = {
                "mape": float(mean_absolute_percentage_error(y, y_pred) * 100),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mse": float(mean_squared_error(y, y_pred)),
                "r2_score": float(r2_score(y, y_pred)),
                "best_params": grid_search.best_params_,
            }
        
        if model_type == "transformer":
            logger.info(
                f"Transformer model trained successfully: "
                f"Best Val Loss={metrics.get('best_val_loss', 0):.4f}"
            )
        else:
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
        elif model_type == "transformer":
            # Create transformer model with hyperparameters
            input_size = len(X.columns)

            model = TransformerPriceForecastModel(
                input_size=input_size,
                hidden_size=hyperparameters.get("hidden_size", 256),
                num_layers=hyperparameters.get("num_layers", 6),
                num_heads=hyperparameters.get("num_heads", 8),
                dropout=hyperparameters.get("dropout", 0.1),
                learning_rate=hyperparameters.get("learning_rate", 1e-4),
                batch_size=hyperparameters.get("batch_size", 32),
                max_epochs=hyperparameters.get("max_epochs", 50),
            )

            # For transformer models, use simple train/validation split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Convert to numpy arrays for transformer training
            X_train_np = X_train.values.astype(np.float32)
            y_train_np = y_train.values.astype(np.float32)
            X_val_np = X_val.values.astype(np.float32)
            y_val_np = y_val.values.astype(np.float32)

            # Train model
            model.fit(X_train_np, y_train_np, X_val_np, y_val_np)

            # Calculate metrics on validation set
            y_pred = model.predict(X_val_np)
            y_true = y_val_np

            metrics = {
                "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mse": float(mean_squared_error(y_true, y_pred)),
                "r2_score": float(r2_score(y_true, y_pred)),
                "best_val_loss": model.training_metrics.get("best_val_loss", 0.0),
                "best_params": hyperparameters,
            }

            return model, metrics
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    async def _train_transformer_model(
        self,
        instrument_id: str,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Dict[str, Any],
    ) -> Tuple[TransformerPriceForecastModel, Dict[str, float]]:
        """Train transformer model with hyperparameter tuning."""
        logger.info(f"Training transformer model for {instrument_id}")

        # Use simple parameter combinations for transformer hyperparameter tuning
        best_model = None
        best_metrics = None
        best_score = float('inf')

        # Try a few parameter combinations
        for hidden_size in hyperparameters.get("hidden_size", [256]):
            for num_layers in hyperparameters.get("num_layers", [6]):
                for dropout in hyperparameters.get("dropout", [0.1]):
                    for lr in hyperparameters.get("learning_rate", [1e-4]):

                        logger.info(f"Trying transformer params: hidden={hidden_size}, layers={num_layers}, dropout={dropout}, lr={lr}")

                        model = TransformerPriceForecastModel(
                            input_size=len(X.columns),
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            learning_rate=lr,
                            max_epochs=30,  # Reduced for hyperparameter search
                        )

                        # Simple train/validation split
                        split_idx = int(len(X) * 0.8)
                        X_train, X_val = X[:split_idx], X[split_idx:]
                        y_train, y_val = y[:split_idx], y[split_idx:]

                        # Convert to numpy arrays
                        X_train_np = X_train.values.astype(np.float32)
                        y_train_np = y_train.values.astype(np.float32)
                        X_val_np = X_val.values.astype(np.float32)
                        y_val_np = y_val.values.astype(np.float32)

                        # Train model
                        model.fit(X_train_np, y_train_np, X_val_np, y_val_np)

                        # Evaluate on validation set
                        y_pred = model.predict(X_val_np)
                        val_loss = mean_squared_error(y_val_np, y_pred)

                        if val_loss < best_score:
                            best_score = val_loss
                            best_model = model
                            best_metrics = {
                                "mape": float(mean_absolute_percentage_error(y_val_np, y_pred) * 100),
                                "rmse": float(np.sqrt(mean_squared_error(y_val_np, y_pred))),
                                "mse": float(mean_squared_error(y_val_np, y_pred)),
                                "r2_score": float(r2_score(y_val_np, y_pred)),
                                "best_val_loss": model.training_metrics.get("best_val_loss", 0.0),
                                "best_params": {
                                    "hidden_size": hidden_size,
                                    "num_layers": num_layers,
                                    "dropout": dropout,
                                    "learning_rate": lr,
                                },
                            }

        if best_model is None:
            raise ValueError("Failed to train transformer model")

        logger.info(f"Best transformer model: Val Loss={best_score:.4f}")
        return best_model, best_metrics

