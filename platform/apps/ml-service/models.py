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

# Transformer model imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

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
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.version = version
        self.model_type = model_type
        self.instrument_id = instrument_id
        self.metrics = metrics
        self.trained_date = datetime.utcnow()
        self.is_active = False
        self.metadata = metadata or {}
    
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
                    "metadata": self.metadata,
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
            metadata=data.get("metadata", {}),
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
        metadata: Optional[Dict[str, Any]] = None,
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
            metadata=metadata,
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
    
    def get_all_instruments(self) -> List[str]:
        """Get list of all instruments that have models."""
        return list(self.models.keys())

    def get_active_model(self, instrument_id: str) -> Optional[PriceForecastModel]:
        """Get the currently active model for an instrument."""
        if instrument_id not in self.models:
            return None

        for model in self.models[instrument_id].values():
            if model.is_active:
                return model

        # If no model is explicitly active, return the most recent
        if self.models[instrument_id]:
            return max(
                self.models[instrument_id].values(),
                key=lambda m: m.trained_date
            )

        return None

    def count(self) -> int:
        """Total number of models loaded."""
        return sum(len(models) for models in self.models.values())


class TransformerPriceForecaster(nn.Module):
    """Transformer-based price forecasting model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

        # Initialize parameters
        self._init_parameters()

    def _create_positional_encoding(self, hidden_size: int, max_len: int = 5000):
        """Create sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-np.log(10000.0) / hidden_size))

        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, max_len, hidden_size)

    def _init_parameters(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        batch_size, seq_len, _ = x.shape

        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, hidden_size)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_size)

        # Use the last timestep for prediction
        x = x[:, -1, :]  # (batch_size, hidden_size)

        # Output projection
        output = self.output_projection(x)  # (batch_size, output_size)

        return output.squeeze(-1)  # (batch_size,)


class TransformerPriceForecastModel:
    """PyTorch Lightning wrapper for transformer price forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 100,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        # Initialize model
        self.model = TransformerPriceForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Training metrics
        self.training_metrics = {}
        self.best_val_loss = float('inf')

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the transformer model."""
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            val_dataset = TimeSeriesDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize Lightning module
        lightning_model = TransformerLightningModule(
            model=self.model,
            learning_rate=self.learning_rate,
        )

        # Train model
        trainer = Trainer(
            max_epochs=self.max_epochs,
            enable_checkpointing=False,
            logger=False,  # Disable logging for API usage
            enable_progress_bar=False,
        )

        trainer.fit(lightning_model, train_loader, val_loader)

        # Update model with trained weights
        self.model = lightning_model.model

        # Calculate training metrics
        self.training_metrics = {
            "final_train_loss": lightning_model.train_losses[-1] if lightning_model.train_losses else 0.0,
            "final_val_loss": lightning_model.val_losses[-1] if lightning_model.val_losses else 0.0,
            "best_val_loss": lightning_model.best_val_loss,
        }

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)

            # Add batch dimension if needed
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)

            predictions = self.model(X_tensor)
            return predictions.numpy()

    def predict_interval(self, X: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals using dropout uncertainty."""
        self.model.train()  # Enable dropout for uncertainty estimation

        # Monte Carlo dropout for uncertainty estimation
        num_samples = 100
        predictions = []

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)

            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)

            for _ in range(num_samples):
                pred = self.model(X_tensor)
                predictions.append(pred.numpy())

        predictions = np.array(predictions)  # (num_samples, batch_size)

        # Calculate percentiles for confidence intervals
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(predictions, lower_percentile, axis=0)
        ci_upper = np.percentile(predictions, upper_percentile, axis=0)

        # Get point predictions (mean)
        point_pred = np.mean(predictions, axis=0)

        return ci_lower, ci_upper

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_type": "transformer",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "training_metrics": self.training_metrics,
        }


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class TransformerLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for transformer model."""

    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log('train_loss', loss, prog_bar=True)
        self.train_losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log('val_loss', loss, prog_bar=True)
        self.val_losses.append(loss.item())

        if loss < self.best_val_loss:
            self.best_val_loss = loss.item()

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
