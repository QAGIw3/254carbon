"""
Model training with hyperparameter tuning.
"""
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Transformer model imports
from deep_learning import MultiCommodityTransformer
from models import TransformerPriceForecastModel
from multimodal_dataset import MultiModalDataset

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
        self.multimodal_defaults: Dict[str, Any] = {
            "seq_len": 64,
            "forecast_horizons": [1, 7, 30],
            "batch_size": 8,
            "epochs": 30,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 4,
            "cross_commodity_layers": 2,
            "d_ff": 1024,
            "dropout": 0.1,
            "grad_clip": 1.0,
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

    async def train_multimodal_transformer(
        self,
        feature_bundle: Dict[str, Dict[str, Any]],
        *,
        hyperparameters: Optional[Dict[str, Any]] = None,
        head_mask_config: Optional[Dict[str, List[int]]] = None,
    ) -> Tuple[MultiCommodityTransformer, Dict[str, Any], Dict[str, Any]]:
        """Train enhanced multimodal transformer on aligned feature bundle."""

        if not feature_bundle:
            raise ValueError("feature_bundle cannot be empty for multimodal training")

        params = dict(self.multimodal_defaults)
        if hyperparameters:
            params.update(hyperparameters)

        horizons = sorted(int(h) for h in params["forecast_horizons"])

        dataset = MultiModalDataset(
            feature_bundle,
            seq_len=int(params["seq_len"]),
            forecast_horizons=horizons,
        )

        total_samples = len(dataset)
        if total_samples < 2:
            raise ValueError("Not enough samples to train multimodal transformer")

        train_size = max(1, int(total_samples * 0.8))
        val_size = total_samples - train_size
        if val_size == 0 and total_samples > 1:
            train_size = total_samples - 1
            val_size = 1

        train_subset = Subset(dataset, range(0, train_size))
        val_subset = Subset(dataset, range(train_size, train_size + val_size)) if val_size > 0 else None

        train_loader = DataLoader(
            train_subset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
            collate_fn=MultiModalDataset.collate_fn,
        )
        val_loader = (
            DataLoader(
                val_subset,
                batch_size=int(params["batch_size"]),
                shuffle=False,
                collate_fn=MultiModalDataset.collate_fn,
            )
            if val_subset is not None
            else None
        )

        commodity_modalities = {
            commodity: {
                modality: frame.shape[1]
                for modality, frame in payload.get("modalities", {}).items()
            }
            for commodity, payload in feature_bundle.items()
        }
        commodity_groups = {
            commodity: payload.get("commodity_group", "unknown")
            for commodity, payload in feature_bundle.items()
        }
        instrument_map = {
            payload.get("instrument_id", commodity): commodity
            for commodity, payload in feature_bundle.items()
        }

        model = MultiCommodityTransformer(
            commodity_modalities=commodity_modalities,
            commodity_groups=commodity_groups,
            d_model=int(params["d_model"]),
            num_heads=int(params["num_heads"]),
            num_layers=int(params["num_layers"]),
            cross_commodity_layers=int(params.get("cross_commodity_layers", 2)),
            d_ff=int(params["d_ff"]),
            dropout=float(params["dropout"]),
            forecast_horizons=horizons,
            head_mask_config=head_mask_config,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        grad_clip = float(params.get("grad_clip", 1.0))

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        history: List[Dict[str, float]] = []

        for epoch in range(int(params["epochs"])):
            model.train()
            train_loss_total = 0.0
            train_batches = 0

            for batch in train_loader:
                inputs, masks, targets = self._prepare_multimodal_batch(batch, device)
                optimizer.zero_grad()
                outputs = model(inputs, commodity_masks=masks)
                loss = self._multimodal_loss(outputs["forecasts"], targets, horizons)
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                train_loss_total += loss.item()
                train_batches += 1

            train_loss = train_loss_total / max(train_batches, 1)

            model.eval()
            val_loss = train_loss
            if val_loader is not None:
                val_loss_total = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, masks, targets = self._prepare_multimodal_batch(batch, device)
                        outputs = model(inputs, commodity_masks=masks)
                        loss = self._multimodal_loss(outputs["forecasts"], targets, horizons)
                        val_loss_total += loss.item()
                        val_batches += 1
                val_loss = val_loss_total / max(val_batches, 1)

            scheduler.step(val_loss)
            history.append({"epoch": epoch + 1, "train_loss": float(train_loss), "val_loss": float(val_loss)})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        model = model.to(torch.device("cpu"))
        device = torch.device("cpu")

        aggregate_metrics, per_commodity_metrics = self._evaluate_multimodal_model(
            model,
            val_loader or train_loader,
            device,
            horizons,
        )

        sample_loader = val_loader or train_loader
        sample_batch = next(iter(sample_loader))
        inputs, masks, _ = self._prepare_multimodal_batch(sample_batch, device)
        with torch.no_grad():
            fusion_snapshot_raw = model(
                inputs,
                commodity_masks=masks,
                return_attentions=True,
            )
        fusion_snapshot = {
            commodity: {
                modality: float(weight.detach().cpu())
                for modality, weight in weights.items()
            }
            for commodity, weights in fusion_snapshot_raw["fusion_gates"].items()
        }

        metrics: Dict[str, Any] = {
            "val_loss": float(best_val_loss),
            "aggregate": aggregate_metrics,
            "per_commodity": per_commodity_metrics,
            "training_history": history,
        }

        artifacts: Dict[str, Any] = {
            "commodity_modalities": commodity_modalities,
            "commodity_groups": commodity_groups,
            "instrument_map": instrument_map,
            "fusion_snapshot": fusion_snapshot,
            "hyperparameters": params,
            "forecast_horizons": horizons,
            "seq_len": int(params["seq_len"]),
        }
        if head_mask_config:
            artifacts["head_mask_config"] = head_mask_config

        model.eval()
        return model, metrics, artifacts

    def _prepare_multimodal_batch(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        masks: Dict[str, Dict[str, torch.Tensor]] = {}
        targets: Dict[str, torch.Tensor] = {}

        for commodity, payload in batch["commodities"].items():
            modality_inputs = {
                modality: tensor.to(device)
                for modality, tensor in payload["modalities"].items()
            }
            modality_masks = {
                modality: tensor.to(device)
                for modality, tensor in payload["masks"].items()
            }
            inputs[commodity] = modality_inputs
            masks[commodity] = modality_masks
            targets[commodity] = payload["targets"].to(device)

        return inputs, masks, targets

    def _multimodal_loss(
        self,
        forecasts: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
        targets: Dict[str, torch.Tensor],
        horizons: List[int],
    ) -> torch.Tensor:
        loss = 0.0
        count = 0
        for commodity, horizon_outputs in forecasts.items():
            target_tensor = targets[commodity]
            for idx, horizon in enumerate(horizons):
                pred = horizon_outputs[horizon]
                mean = pred["mean"]
                std = pred["std"].clamp_min(1e-6)
                var = std ** 2
                target = target_tensor[:, idx]
                nll = 0.5 * torch.log(2 * math.pi * var) + 0.5 * ((target - mean) ** 2) / var
                loss = loss + nll.mean()
                count += 1
        return loss / max(count, 1)

    def _evaluate_multimodal_model(
        self,
        model: MultiCommodityTransformer,
        loader: DataLoader,
        device: torch.device,
        horizons: List[int],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        per_commodity: Dict[str, Dict[str, float]] = {}
        aggregate_accumulator = {"ape_sum": 0.0, "se_sum": 0.0, "count": 0}

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in loader:
                inputs, masks, targets = self._prepare_multimodal_batch(batch, device)
                outputs = model(inputs, commodity_masks=masks)
                for commodity, horizon_outputs in outputs["forecasts"].items():
                    stats = per_commodity.setdefault(
                        commodity,
                        {"ape_sum": 0.0, "se_sum": 0.0, "count": 0},
                    )
                    target_tensor = targets[commodity]
                    for idx, horizon in enumerate(horizons):
                        mean = horizon_outputs[horizon]["mean"]
                        target = target_tensor[:, idx]
                        denom = torch.clamp(target.abs(), min=1e-6)
                        stats["ape_sum"] += torch.abs((target - mean) / denom).sum().item()
                        stats["se_sum"] += ((target - mean) ** 2).sum().item()
                        stats["count"] += target.numel()

        per_commodity_metrics: Dict[str, Dict[str, float]] = {}
        for commodity, stats in per_commodity.items():
            count = max(stats["count"], 1)
            mape = (stats["ape_sum"] / count) * 100
            rmse = math.sqrt(stats["se_sum"] / count)
            per_commodity_metrics[commodity] = {
                "mape": float(mape),
                "rmse": float(rmse),
                "count": int(stats["count"]),
            }
            aggregate_accumulator["ape_sum"] += stats["ape_sum"]
            aggregate_accumulator["se_sum"] += stats["se_sum"]
            aggregate_accumulator["count"] += stats["count"]

        aggregate_metrics: Dict[str, float] = {}
        if aggregate_accumulator["count"] > 0:
            aggregate_metrics = {
                "mape": float(
                    (aggregate_accumulator["ape_sum"] / aggregate_accumulator["count"]) * 100
                ),
                "rmse": float(
                    math.sqrt(aggregate_accumulator["se_sum"] / aggregate_accumulator["count"])
                ),
                "count": int(aggregate_accumulator["count"]),
            }

        return aggregate_metrics, per_commodity_metrics
