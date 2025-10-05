"""
Machine learning-based congestion prediction for LMP decomposition.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import asyncio
from clickhouse_driver import Client

logger = logging.getLogger(__name__)


class CongestionPredictor:
    """ML-based congestion forecasting using historical patterns and binding constraints."""

    def __init__(self):
        self.ch_client = Client(host="clickhouse", port=9000)

        # Model configuration
        self.models = {
            "rf": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            "gbm": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }

        self.scaler = StandardScaler()
        self.model_performance = {}
        self.trained_models = {}

        # Load pre-trained models if available
        self._load_models()

    def _load_models(self):
        """Load pre-trained models from disk."""
        model_dir = "/app/models"
        if os.path.exists(model_dir):
            for iso in ["PJM", "MISO", "ERCOT", "CAISO"]:
                for model_type in self.models.keys():
                    model_path = f"{model_dir}/congestion_{model_type}_{iso}.pkl"
                    if os.path.exists(model_path):
                        try:
                            self.trained_models[f"{model_type}_{iso}"] = joblib.load(model_path)
                            logger.info(f"Loaded {model_type} model for {iso}")
                        except Exception as e:
                            logger.warning(f"Failed to load {model_path}: {e}")

    async def get_historical_congestion(
        self,
        node_id: str,
        lookback_days: int = 90,
        iso: str = "PJM"
    ) -> pd.DataFrame:
        """Get historical congestion data for ML training."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        query = """
        SELECT
            event_time as timestamp,
            instrument_id as node_id,
            value as lmp
        FROM ch.market_price_ticks
        WHERE instrument_id = %(node_id)s
          AND event_time BETWEEN %(start)s AND %(end)s
          AND price_type IN ('trade', 'settle')
        ORDER BY event_time
        """

        result = self.ch_client.execute(
            query,
            {
                "node_id": node_id,
                "start": start_date,
                "end": end_date,
            },
        )

        df = pd.DataFrame(result, columns=["timestamp", "node_id", "lmp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Calculate congestion component (deviation from hub)
        hub_id = self._get_reference_hub(iso)
        hub_prices = await self._get_hub_prices(hub_id, start_date, end_date, iso)

        df = df.join(hub_prices, how="inner", rsuffix="_hub")
        df["congestion"] = df["lmp"] - df["lmp_hub"]

        return df[["congestion"]].dropna()

    async def _get_hub_prices(
        self,
        hub_id: str,
        start_date: datetime,
        end_date: datetime,
        iso: str
    ) -> pd.Series:
        """Get hub prices for reference."""
        query = """
        SELECT
            event_time as timestamp,
            value as lmp
        FROM ch.market_price_ticks
        WHERE instrument_id = %(hub_id)s
          AND event_time BETWEEN %(start)s AND %(end)s
          AND price_type IN ('trade', 'settle')
        ORDER BY event_time
        """

        result = self.ch_client.execute(
            query,
            {
                "hub_id": hub_id,
                "start": start_date,
                "end": end_date,
            },
        )

        df = pd.DataFrame(result, columns=["timestamp", "lmp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        return df["lmp"]

    def _get_reference_hub(self, iso: str) -> str:
        """Get reference hub for the ISO."""
        hubs = {
            "PJM": "PJM.HUB.WEST",
            "MISO": "MISO.HUB.INDIANA",
            "ERCOT": "ERCOT.HUB.NORTH",
            "CAISO": "CAISO.HUB.SP15",
        }
        return hubs.get(iso, hubs["PJM"])

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from congestion time series."""
        features = pd.DataFrame(index=df.index)

        # Time-based features
        features["hour"] = df.index.hour
        features["day_of_week"] = df.index.dayofweek
        features["month"] = df.index.month
        features["quarter"] = df.index.quarter
        features["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

        # Lag features (congestion from previous hours)
        for lag in [1, 2, 3, 6, 12, 24]:
            features[f"congestion_lag_{lag}h"] = df["congestion"].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            features[f"congestion_mean_{window}d"] = df["congestion"].rolling(window).mean()
            features[f"congestion_std_{window}d"] = df["congestion"].rolling(window).std()
            features[f"congestion_max_{window}d"] = df["congestion"].rolling(window).max()
            features[f"congestion_min_{window}d"] = df["congestion"].rolling(window).min()

        # Volatility features
        features["congestion_volatility"] = df["congestion"].rolling(24).std()

        # Trend features
        features["congestion_trend"] = df["congestion"] - df["congestion"].shift(24)

        return features.dropna()

    async def train_model(
        self,
        node_id: str,
        iso: str,
        lookback_days: int = 365,
        test_size: float = 0.2,
        model_type: str = "rf"
    ) -> Dict[str, Any]:
        """Train ML model for congestion prediction."""
        logger.info(f"Training {model_type} model for {node_id} in {iso}")

        # Get historical data
        df = await self.get_historical_congestion(node_id, lookback_days, iso)

        if len(df) < 100:
            raise ValueError(f"Insufficient data for {node_id}: {len(df)} samples")

        # Create features and target
        features = self._create_features(df)
        target = df["congestion"].shift(-1).dropna()  # Predict next hour

        # Align features and target
        features = features.loc[target.index]

        if len(features) < 50:
            raise ValueError(f"Insufficient aligned data for {node_id}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        model = self.models[model_type]
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Save model
        model_key = f"{model_type}_{iso}_{node_id}"
        self.trained_models[model_key] = model

        # Save to disk
        os.makedirs("/app/models", exist_ok=True)
        joblib.dump(model, f"/app/models/congestion_{model_type}_{iso}_{node_id}.pkl")

        performance = {
            "node_id": node_id,
            "iso": iso,
            "model_type": model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "feature_count": len(features.columns),
            "training_date": datetime.now().isoformat()
        }

        self.model_performance[model_key] = performance

        logger.info(f"Trained model {model_key}: MAE={mae:.4f}, RMSE={rmse:.4f}")

        return performance

    async def predict_congestion(
        self,
        node_id: str,
        forecast_date: datetime,
        hours_ahead: int = 24,
        iso: str = "PJM",
        model_type: str = "rf"
    ) -> Dict[str, Any]:
        """Predict future congestion using trained ML model."""
        model_key = f"{model_type}_{iso}_{node_id}"

        if model_key not in self.trained_models:
            # Try to train model on-the-fly if not available
            await self.train_model(node_id, iso, model_type=model_type)

        if model_key not in self.trained_models:
            raise ValueError(f"No trained model available for {node_id}")

        model = self.trained_models[model_key]

        # Create features for prediction
        # Use recent historical data to create features for forecast
        end_date = forecast_date - timedelta(hours=1)
        start_date = end_date - timedelta(days=30)  # Need 30 days for features

        df = await self.get_historical_congestion(node_id, 30, iso)

        if df.empty:
            raise ValueError(f"No historical data available for {node_id}")

        # Create features for the forecast period
        forecast_features = []
        base_time = forecast_date

        for hour in range(hours_ahead):
            current_time = base_time + timedelta(hours=hour)

            # Create feature row similar to training
            feature_row = {
                "hour": current_time.hour,
                "day_of_week": current_time.weekday(),
                "month": current_time.month,
                "quarter": (current_time.month - 1) // 3 + 1,
                "is_weekend": 1 if current_time.weekday() in [5, 6] else 0,
            }

            # Add recent congestion values (use last known values)
            for lag in [1, 2, 3, 6, 12, 24]:
                # Use the most recent available congestion value
                if not df.empty:
                    feature_row[f"congestion_lag_{lag}h"] = df["congestion"].iloc[-1]
                else:
                    feature_row[f"congestion_lag_{lag}h"] = 0.0

            # Rolling statistics (use recent averages)
            for window in [7, 14, 30]:
                if len(df) >= window:
                    feature_row[f"congestion_mean_{window}d"] = df["congestion"].tail(window).mean()
                    feature_row[f"congestion_std_{window}d"] = df["congestion"].tail(window).std()
                    feature_row[f"congestion_max_{window}d"] = df["congestion"].tail(window).max()
                    feature_row[f"congestion_min_{window}d"] = df["congestion"].tail(window).min()
                else:
                    feature_row[f"congestion_mean_{window}d"] = 0.0
                    feature_row[f"congestion_std_{window}d"] = 0.0
                    feature_row[f"congestion_max_{window}d"] = 0.0
                    feature_row[f"congestion_min_{window}d"] = 0.0

            forecast_features.append(feature_row)

        # Convert to DataFrame and scale
        feature_df = pd.DataFrame(forecast_features)
        feature_scaled = self.scaler.transform(feature_df)

        # Make predictions
        predictions = model.predict(feature_scaled)

        # Create forecast response
        forecast = []
        for i, pred in enumerate(predictions):
            forecast_time = base_time + timedelta(hours=i)
            forecast.append({
                "timestamp": forecast_time.isoformat(),
                "predicted_congestion": float(pred),
                "confidence": 0.8  # Placeholder confidence score
            })

        return {
            "node_id": node_id,
            "iso": iso,
            "forecast_date": forecast_date.isoformat(),
            "hours_ahead": hours_ahead,
            "model_type": model_type,
            "forecast": forecast,
            "model_performance": self.model_performance.get(model_key, {})
        }

    async def get_model_performance(
        self,
        node_id: str,
        iso: str,
        model_type: str = "rf"
    ) -> Dict[str, Any]:
        """Get performance metrics for a trained model."""
        model_key = f"{model_type}_{iso}_{node_id}"
        return self.model_performance.get(model_key, {})

    async def retrain_models_periodically(self):
        """Periodic model retraining for all nodes."""
        # This would be called by a background scheduler
        # For now, just log that it should be implemented
        logger.info("Model retraining scheduled - implement with APScheduler or similar")
