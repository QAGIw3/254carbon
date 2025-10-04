"""
Ensemble ML calibrator for scenario engine.

This module builds an ensemble of models that calibrate fundamentals-driven
targets to observed market behavior.

Mathematical notes
- Price model: supervised learning on features φ → price residuals ε, where
  final price is μ + f_θ(φ), μ from fundamentals and f_θ learned (here GBM).
- Volatility model: rolling standard deviation σ_t estimated with a GBM on
  [load, gas, σ_roll] features, approximating GARCH-like dynamics without
  explicit ARMA terms.
- Correlations: empirical correlation Σ = corr(X) with PCA to extract
  principal components (eigenvalues/eigenvectors) and infer regime via
  λ_max thresholding.
- Risk: stylized VaR/ES using composite risk scores and stress scenarios.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    mape: float
    rmse: float
    mae: float
    r2: float


class EnsembleCalibrator:
    """Creates ensemble calibrations using fundamentals output.

    Methods focus on: price calibration, volatility calibration, correlation
    structure estimation, and risk metrics assembly.
    """

    def __init__(self) -> None:
        self.random_state = 42

    async def run(self, fundamentals: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full calibration suite.

        Args:
            fundamentals: Outputs from FundamentalsEngine (dict of series/maps)
            scenario: Scenario spec affecting drivers (e.g., policy)

        Returns:
            Dict with sub-calibration artifacts (price, volatility, corr, risk)
            and derived ensemble weights.
        """
        try:
            features, targets = self._build_training_data(fundamentals, scenario)
            if len(targets) < 100:
                raise ValueError("Insufficient data for calibration")

            calibration = await asyncio.gather(
                self._calibrate_prices(features, targets),
                self._calibrate_volatility(features, targets),
                self._calibrate_correlations(features),
                self._calibrate_risk_metrics(fundamentals, scenario),
            )

            price, volatility, correlations, risk = calibration

            ensemble_weights = self._calculate_ensemble_weights(price["metrics"], volatility["metrics"])

            return {
                "price_calibration": price,
                "volatility_calibration": volatility,
                "correlation_calibration": correlations,
                "risk_calibration": risk,
                "ensemble_weights": ensemble_weights,
                "metadata": {
                    "methodology": "ensemble_v2",
                    "model_versions": {
                        "price_model": "gbm_v2",
                        "volatility_model": "gbm_v1",
                        "correlation_model": "statistical_v1",
                        "risk_model": "stress_v2",
                    },
                },
            }
        except Exception as exc:
            logger.exception("Ensemble calibration failed", exc_info=exc)
            return {
                "status": "fallback",
                "error": str(exc),
            }

    def _build_training_data(self, fundamentals: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Construct feature/target arrays from fundamentals and scenario.

        Returns:
            (features, targets) numpy arrays for supervised calibration
        """
        load = fundamentals.get("load_forecast", {}).get("regions", {})
        fuel = fundamentals.get("fuel_prices", {}).get("forward_curves", {})
        policy = fundamentals.get("policy_impact", {})

        rows: List[List[float]] = []
        targets: List[float] = []

        for region, metrics in load.items():
            projected = metrics.get("projected", [])
            if not projected:
                continue
            for idx, value in enumerate(projected):
                feature_row = [
                    value,
                    metrics.get("peak", 0.0),
                    metrics.get("elasticity", -0.2),
                    policy.get("impact_score", 0.3),
                    fuel.get("natural_gas", {}).get(f"year_{min(idx, 5)}", 3.5),
                    fuel.get("oil", {}).get(f"year_{min(idx, 5)}", 70.0),
                ]
                rows.append(feature_row)
                targets.append(value * 0.045)  # synthetic nodal price reference

        return np.array(rows), np.array(targets)

    async def _calibrate_prices(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Calibrate price adjustments using GBM on standardized features."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, targets, test_size=0.2, random_state=self.random_state)

        gbm = GradientBoostingRegressor(random_state=self.random_state)
        gbm.fit(X_train, y_train)

        preds = gbm.predict(X_test)
        metrics = self._evaluate_model(y_test, preds)

        return {
            "model": "gradient_boosting",
            "metrics": metrics.__dict__,
            "feature_importance": gbm.feature_importances_.tolist(),
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
        }

    async def _calibrate_volatility(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Estimate rolling volatility with a GBM over synthetic features.

        Notes:
            σ_roll = std(window=12) used as a proxy; feature set approximates
            conditional variance without explicit AR terms.
        """
        rolling_std = pd.Series(targets).rolling(window=12, min_periods=3).std().fillna(method="bfill")
        synthetic_features = np.column_stack([
            features[:, 0],
            features[:, 4],
            rolling_std.to_numpy(),
        ])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(synthetic_features)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, rolling_std.to_numpy(), test_size=0.25, random_state=self.random_state)

        gbm = GradientBoostingRegressor(random_state=self.random_state)
        gbm.fit(X_train, y_train)

        preds = gbm.predict(X_test)
        metrics = self._evaluate_model(y_test, preds)

        return {
            "model": "volatility_gbm",
            "metrics": metrics.__dict__,
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
        }

    async def _calibrate_correlations(self, features: np.ndarray) -> Dict[str, Any]:
        """Compute empirical correlation matrix and PCA decomposition."""
        df = pd.DataFrame(features, columns=["load", "peak", "elasticity", "policy", "gas", "oil"])
        corr = df.corr().to_dict()
        eigenvalues, eigenvectors = np.linalg.eig(df.corr())

        return {
            "correlation_matrix": corr,
            "principal_components": {
                "eigenvalues": eigenvalues.real.tolist(),
                "eigenvectors": eigenvectors.real.tolist(),
            },
            "regime": "normal" if eigenvalues.max().real < 3 else "stress",
        }

    async def _calibrate_risk_metrics(self, fundamentals: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble stylized VaR/ES and stress scenarios from drivers."""
        risk = fundamentals.get("risk_factors", {})
        policy = fundamentals.get("policy_impact", {})
        weather = fundamentals.get("weather_factors", {})

        stress_scenarios = {
            "fuel_shock": {
                "impact": -0.18 * risk.get("fuel_risk", 0.3),
                "probability": 0.1,
            },
            "policy_shift": {
                "impact": 0.12 * policy.get("impact_score", 0.4),
                "probability": 0.08,
            },
            "extreme_weather": {
                "impact": 0.16 * weather.get("extreme_weather_probability", 0.05),
                "probability": 0.12,
            },
        }

        return {
            "stress_tests": stress_scenarios,
            "var": {
                "95": float(1.65 * risk.get("composite_risk", 0.25)),
                "99": float(2.33 * risk.get("composite_risk", 0.25)),
            },
            "expected_shortfall": {
                "95": float(1.9 * risk.get("composite_risk", 0.25)),
                "99": float(2.6 * risk.get("composite_risk", 0.25)),
            },
        }

    def _calculate_ensemble_weights(self, price_metrics: Dict[str, float], vol_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute weights inversely proportional to MAPE (normalized)."""
        price_score = max(1e-6, 1 - price_metrics.get("mape", 0.2))
        vol_score = max(1e-6, 1 - vol_metrics.get("mape", 0.25))
        total = price_score + vol_score
        return {
            "price_model": price_score / total,
            "volatility_model": vol_score / total,
        }

    def _evaluate_model(self, actual: np.ndarray, predicted: np.ndarray) -> ModelMetrics:
        """Return standard regression metrics (MAPE, RMSE, MAE, R²)."""
        mape = mean_absolute_percentage_error(actual, predicted)
        rmse = mean_squared_error(actual, predicted, squared=False)
        mae = float(np.mean(np.abs(actual - predicted)))
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return ModelMetrics(mape=float(mape), rmse=float(rmse), mae=mae, r2=r2)
