"""Regional gas basis modeling."""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..clients.clickhouse import query_dataframe, insert_rows
from ..models import GasBasisModelResult

logger = logging.getLogger(__name__)

FEATURE_METRICS = [
    "pipeline_utilization_pct",
    "storage_deviation_pct",
    "lng_utilization_pct",
    "hdd",
    "cdd",
]


class GasBasisModeler:
    """Simple linear model that maps fundamentals to regional basis."""

    def __init__(self, ridge_lambda: float = 0.1, lookback_days: int = 240):
        self.ridge_lambda = ridge_lambda
        self.lookback_days = lookback_days

    def _load_prices(self, hub: str, as_of: date) -> pd.DataFrame:
        sql = """
            SELECT date, avg_price
            FROM ch.market_price_daily_agg
            WHERE instrument_id = %(hub)s
              AND date <= %(as_of)s
            ORDER BY date DESC
            LIMIT %(limit)s
        """
        df = query_dataframe(sql, {"hub": hub, "as_of": as_of, "limit": self.lookback_days})
        if df.empty:
            logger.warning("No price data for hub %s; synthesizing basis inputs", hub)
            idx = pd.date_range(end=as_of, periods=self.lookback_days, freq="D")
            series = 3.0 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, len(idx)))
            df = pd.DataFrame({"date": idx.date, "avg_price": series})
        return df.set_index("date").sort_index()

    def _load_features(self, entity_id: str, as_of: date) -> pd.DataFrame:
        sql = """
            SELECT date, metric_name, metric_value
            FROM ch.supply_demand_metrics
            WHERE entity_id = %(entity)s
              AND metric_name IN %(metrics)s
              AND date <= %(as_of)s
            ORDER BY date DESC
            LIMIT %(limit)s
        """
        df = query_dataframe(
            sql,
            {
                "entity": entity_id,
                "metrics": tuple(FEATURE_METRICS),
                "as_of": as_of,
                "limit": self.lookback_days * len(FEATURE_METRICS),
            },
        )
        if df.empty:
            logger.warning("Fundamental metrics missing for %s; generating synthetic features", entity_id)
            idx = pd.date_range(end=as_of, periods=self.lookback_days, freq="D")
            fake = {metric: 0.5 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, len(idx))) for metric in FEATURE_METRICS}
            out = pd.DataFrame(fake, index=idx.date)
            return out
        pivot = df.pivot_table(index="date", columns="metric_name", values="metric_value", aggfunc="last")
        return pivot.sort_index()

    def _prepare_dataset(self, hub: str, as_of: date) -> pd.DataFrame:
        hub_prices = self._load_prices(hub, as_of)
        henry_prices = self._load_prices("HENRY", as_of)
        df = hub_prices.join(henry_prices, how="inner", lsuffix="_hub", rsuffix="_henry")
        df["basis"] = df["avg_price_hub"] - df["avg_price_henry"]
        features = self._load_features(hub, as_of)
        dataset = df.join(features, how="left")
        dataset = dataset.dropna(subset=["basis"]).tail(self.lookback_days)
        for metric in FEATURE_METRICS:
            if metric not in dataset.columns:
                dataset[metric] = np.nan
        dataset[FEATURE_METRICS] = dataset[FEATURE_METRICS].fillna(method="ffill").fillna(method="bfill")
        dataset = dataset.dropna()
        return dataset

    def compute(self, hub: str, as_of: date, method: str = "ridge") -> GasBasisModelResult:
        dataset = self._prepare_dataset(hub, as_of)
        if dataset.empty:
            raise ValueError(f"No data available to fit basis model for {hub}")

        y = dataset["basis"].values
        X = dataset[FEATURE_METRICS].values
        X = np.column_stack([np.ones(len(X)), X])
        lambda_identity = self.ridge_lambda * np.eye(X.shape[1])
        beta = np.linalg.pinv(X.T @ X + lambda_identity) @ X.T @ y
        y_pred = X @ beta
        resid = y - y_pred
        ss_total = float(((y - y.mean()) ** 2).sum())
        ss_res = float((resid ** 2).sum())
        r2 = 1 - ss_res / ss_total if ss_total else 0.0

        latest_features = dataset.iloc[-1][FEATURE_METRICS].to_dict()
        latest_basis = float(dataset.iloc[-1]["basis"])
        predicted_basis = float(y_pred[-1])

        coeff_map = {"intercept": float(beta[0])}
        coeff_map.update({feature: float(coef) for feature, coef in zip(FEATURE_METRICS, beta[1:])})

        diagnostics = {
            "r2": r2,
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "coefficients": coeff_map,
            "training_samples": len(dataset),
        }

        result = GasBasisModelResult(
            as_of_date=as_of,
            hub=hub,
            predicted_basis=predicted_basis,
            actual_basis=latest_basis,
            method=method,
            diagnostics=diagnostics,
            feature_snapshot=latest_features,
        )
        return result

    def persist(self, result: GasBasisModelResult) -> None:
        row = {
            "as_of_date": result.as_of_date,
            "hub": result.hub,
            "predicted_basis": result.predicted_basis,
            "actual_basis": result.actual_basis,
            "feature_snapshot": json.dumps(result.feature_snapshot),
            "diagnostics": json.dumps(result.diagnostics),
            "method": result.method,
            "model_version": "v1",
        }
        insert_rows("market_intelligence.gas_basis_models", [row])
