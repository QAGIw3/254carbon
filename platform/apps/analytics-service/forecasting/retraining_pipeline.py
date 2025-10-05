"""
Automated ML model retraining pipeline with performance monitoring.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import requests
from clickhouse_driver import Client

from models import ModelRegistry
from training import ModelTrainer
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Automated model retraining pipeline with performance monitoring.

    Monitors model performance and triggers retraining when performance
    degrades below acceptable thresholds.
    """

    def __init__(self):
        self.ch_client = Client(host="clickhouse", port=9000)
        self.model_registry = ModelRegistry()
        self.trainer = ModelTrainer()
        self.feature_engineer = FeatureEngineer()

        self.prometheus_gateway = os.environ.get("PROMETHEUS_PUSHGATEWAY", "http://prometheus:9091")
        self.alert_webhook = os.environ.get("ALERT_WEBHOOK_URL")

        # Performance thresholds
        self.performance_thresholds = {
            "mape_warning": 0.12,   # 12% MAPE triggers warning
            "mape_critical": 0.20,  # 20% MAPE triggers critical alert
            "retrain_threshold": 0.15,  # Retrain when MAPE > 15%
        }

    async def check_model_performance(self, instrument_id: str) -> Dict[str, float]:
        """
        Check current model performance for an instrument.

        Returns metrics like MAPE, RMSE, R² over recent period.
        """
        try:
            # Get the active model for this instrument
            model_info = self.model_registry.get_active_model(instrument_id)
            if not model_info:
                return {"mape": 1.0, "rmse": 100.0, "r2": 0.0, "status": "no_model"}

            # Get recent predictions and actuals for evaluation
            evaluation_data = await self._get_evaluation_data(
                instrument_id,
                lookback_days=30
            )

            if evaluation_data.empty:
                return {"mape": 0.0, "rmse": 0.0, "r2": 0.0, "status": "no_data"}

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                evaluation_data["predicted"],
                evaluation_data["actual"]
            )

            drift_metrics = self._calculate_drift_metrics(evaluation_data)
            metrics.update(drift_metrics)
            metrics["status"] = self._determine_performance_status(metrics["mape"], drift_metrics)

            await self._emit_prometheus_metrics(instrument_id, metrics)

            if metrics["status"] == "failed":
                await self._send_alert(instrument_id, metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error checking performance for {instrument_id}: {e}")
            return {"mape": 1.0, "rmse": 100.0, "r2": 0.0, "status": "error", "error": str(e)}

    async def retrain_model(
        self,
        instrument_id: str,
        model_type: str = "xgboost",
        retrain_window_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Execute full retraining pipeline for a model.

        1. Gather training data
        2. Feature engineering
        3. Model training and validation
        4. Performance evaluation
        5. Model registration
        """
        logger.info(f"Starting retraining pipeline for {instrument_id}")

        try:
            # Step 1: Get training data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=retrain_window_days)

            logger.info(f"Fetching training data for {instrument_id}")
            training_data = await self.feature_engineer.build_training_dataset(
                instrument_id=instrument_id,
                start_date=start_date,
                end_date=end_date,
            )

            if training_data.empty:
                raise ValueError(f"No training data available for {instrument_id}")

            # Step 2: Train model
            logger.info(f"Training {model_type} model for {instrument_id}")
            model, training_metrics = await self.trainer.train(
                instrument_id=instrument_id,
                data=training_data,
                model_type=model_type,
            )

            # Step 3: Evaluate on recent data
            logger.info(f"Evaluating model performance for {instrument_id}")
            evaluation_metrics = await self._evaluate_model_performance(
                model, instrument_id, model_type
            )

            # Step 4: Register new model
            logger.info(f"Registering new model for {instrument_id}")
            model_version = self.model_registry.register(
                instrument_id=instrument_id,
                model=model,
                metrics=evaluation_metrics,
            )

            # Step 5: Log retraining event
            await self._log_retraining_event(
                instrument_id=instrument_id,
                model_version=model_version,
                training_metrics=training_metrics,
                evaluation_metrics=evaluation_metrics,
            )

            return {
                "instrument_id": instrument_id,
                "model_version": model_version,
                "model_type": model_type,
                "training_metrics": training_metrics,
                "evaluation_metrics": evaluation_metrics,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error in retraining pipeline for {instrument_id}: {e}")
            return {
                "instrument_id": instrument_id,
                "status": "failed",
                "error": str(e),
            }

    async def get_performance_metrics(
        self,
        instrument_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get detailed performance metrics over a time period.
        """
        try:
            # Get evaluation data
            evaluation_data = await self._get_evaluation_data(
                instrument_id, lookback_days=days
            )

            if evaluation_data.empty:
                return {"error": "No evaluation data available"}

            # Calculate metrics by time period
            daily_metrics = []

            for eval_date in evaluation_data.index.date:
                day_data = evaluation_data[evaluation_data.index.date == eval_date]
                if len(day_data) > 0:
                    metrics = self._calculate_performance_metrics(
                        day_data["predicted"],
                        day_data["actual"]
                    )
                    metrics["date"] = eval_date.isoformat()
                    daily_metrics.append(metrics)

            # Overall metrics
            overall_metrics = self._calculate_performance_metrics(
                evaluation_data["predicted"],
                evaluation_data["actual"]
            )

            drift_metrics = self._calculate_drift_metrics(evaluation_data)

            return {
                "overall": overall_metrics,
                "daily": daily_metrics,
                "drift": drift_metrics,
                "evaluation_period_days": days,
            }

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}

    async def _get_evaluation_data(
        self,
        instrument_id: str,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Get recent predictions and actuals for model evaluation.
        """
        try:
            # Query ClickHouse for recent predictions and actuals
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            query = """
            SELECT
                toDate(event_time) as date,
                instrument_id,
                value as actual,
                -- For now, use actual as predicted (in production, join with predictions table)
                value as predicted
            FROM market_price_ticks
            WHERE instrument_id = %(instrument_id)s
              AND toDate(event_time) >= %(start_date)s
              AND toDate(event_time) <= %(end_date)s
            ORDER BY event_time
            """

            result = self.ch_client.execute(
                query,
                {
                    "instrument_id": instrument_id,
                    "start_date": start_date.date(),
                    "end_date": end_date.date(),
                }
            )

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(result, columns=["date", "instrument_id", "actual", "predicted"])
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error getting evaluation data: {e}")
            return pd.DataFrame()

    def _calculate_performance_metrics(
        self,
        predicted: pd.Series,
        actual: pd.Series,
    ) -> Dict[str, float]:
        """Calculate standard forecasting performance metrics."""
        try:
            valid_mask = ~(predicted.isna() | actual.isna())
            predicted = predicted[valid_mask]
            actual = actual[valid_mask]

            if len(predicted) == 0:
                return {"mape": 1.0, "rmse": 100.0, "r2": 0.0, "mae": 0.0}

            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(np.mean((predicted - actual) ** 2))
            mae = np.mean(np.abs(predicted - actual))

            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {
                "mape": mape,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "sample_size": len(predicted),
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"mape": 1.0, "rmse": 100.0, "r2": 0.0, "mae": 0.0}

    def _determine_performance_status(self, mape: float, drift_metrics: Optional[Dict[str, float]] = None) -> str:
        """Determine performance status with drift-awareness."""
        if mape < self.performance_thresholds["mape_warning"]:
            status = "healthy"
        elif mape < self.performance_thresholds["mape_critical"]:
            status = "degraded"
        else:
            status = "failed"

        if drift_metrics:
            ks_stat = drift_metrics.get("ks_statistic", 0.0)
            psi = drift_metrics.get("population_stability_index", 0.0)
            if ks_stat > 0.15 or psi > 0.25:
                status = "failed"
            elif ks_stat > 0.1 or psi > 0.15:
                status = "degraded"

        return status

    async def _evaluate_model_performance(
        self,
        model,
        instrument_id: str,
        model_type: str,
    ) -> Dict[str, float]:
        """Evaluate trained model on recent data."""
        try:
            # Get recent evaluation data
            evaluation_data = await self._get_evaluation_data(
                instrument_id, lookback_days=30
            )

            if evaluation_data.empty:
                return {"mape": 0.0, "rmse": 0.0, "r2": 0.0}

            # For now, use simple evaluation (in production, would use proper model prediction)
            # This is a placeholder - actual implementation would call the model
            metrics = self._calculate_performance_metrics(
                evaluation_data["predicted"],
                evaluation_data["actual"]
            )

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {"mape": 1.0, "rmse": 100.0, "r2": 0.0}

    async def _log_retraining_event(
        self,
        instrument_id: str,
        model_version: str,
        training_metrics: Dict[str, Any],
        evaluation_metrics: Dict[str, Any],
    ) -> None:
        """Log retraining event for audit trail."""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "model_retraining",
                "instrument_id": instrument_id,
                "model_version": model_version,
                "training_metrics": training_metrics,
                "evaluation_metrics": evaluation_metrics,
            }

            # In production, would store in audit log table
            logger.info(f"Retraining completed for {instrument_id}: {log_entry}")

        except Exception as e:
            logger.error(f"Error logging retraining event: {e}")

    async def schedule_automatic_retraining(
        self,
        check_interval_hours: int = 24,
    ) -> None:
        """
        Start automatic retraining scheduler.

        Runs periodically to check model performance and trigger retraining.
        """
        logger.info(f"Starting automatic retraining scheduler (interval: {check_interval_hours}h)")

        while True:
            try:
                # Get all instruments that need monitoring
                all_instruments = await self.model_registry.get_all_instruments()

                instruments_needing_retraining = []

                for instrument_id in all_instruments:
                    performance = await self.check_model_performance(instrument_id)

                    if performance.get("status") in ["degraded", "failed"]:
                        instruments_needing_retraining.append(instrument_id)

                if instruments_needing_retraining:
                    logger.info(f"Found {len(instruments_needing_retraining)} instruments needing retraining")

                    # Trigger retraining for instruments that need it
                    retrain_results = []
                    for instrument_id in instruments_needing_retraining[:5]:  # Limit batch size
                        result = await self.retrain_model(instrument_id)
                        retrain_results.append(result)

                    logger.info(f"Completed retraining batch: {len(retrain_results)} models updated")
                else:
                    logger.info("All models performing within acceptable thresholds")

            except Exception as e:
                logger.error(f"Error in automatic retraining check: {e}")

            # Wait for next check
            await asyncio.sleep(check_interval_hours * 3600)
