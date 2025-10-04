"""Weather impact regression for HDD/CDD betas."""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import List

import numpy as np
import pandas as pd

from ..clients.clickhouse import query_dataframe, insert_rows
from ..models import WeatherImpactCoefficient

logger = logging.getLogger(__name__)


DEFAULT_WINDOW = 120


class WeatherImpactAnalyzer:
    """Rolling OLS estimation of HDD/CDD price sensitivity."""

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window

    def _load_prices(self, entity_id: str, as_of: date) -> pd.DataFrame:
        sql = """
            SELECT
                date,
                avg_price AS price
            FROM ch.market_price_daily_agg
            WHERE instrument_id = %(entity)s
              AND date <= %(as_of)s
            ORDER BY date
            LIMIT 1000
        """
        df = query_dataframe(sql, {"entity": entity_id, "as_of": as_of})
        if df.empty:
            logger.warning("No price history for %s; synthesizing series", entity_id)
            idx = pd.date_range(end=as_of, periods=365, freq="D")
            base = 3.0 + 0.5 * np.sin(np.linspace(0, 6 * np.pi, len(idx)))
            noise = np.random.normal(0, 0.15, len(idx))
            df = pd.DataFrame({"date": idx.date, "price": base + noise})
        return df

    def _load_degree_days(self, entity_id: str, as_of: date) -> pd.DataFrame:
        sql = """
            SELECT
                date,
                metric_name,
                metric_value
            FROM ch.supply_demand_metrics
            WHERE entity_id = %(entity)s
              AND metric_name IN ('hdd', 'cdd')
              AND date <= %(as_of)s
            ORDER BY date
            LIMIT 1000
        """
        df = query_dataframe(sql, {"entity": entity_id, "as_of": as_of})
        if df.empty:
            logger.warning("No HDD/CDD metrics for %s; synthesizing features", entity_id)
            idx = pd.date_range(end=as_of, periods=365, freq="D")
            temps = 65 + 20 * np.sin(np.linspace(0, 6 * np.pi, len(idx)))
            hdd = np.maximum(65 - temps, 0)
            cdd = np.maximum(temps - 65, 0)
            df = pd.DataFrame({
                "date": np.concatenate((idx.date, idx.date)),
                "metric_name": ["hdd"] * len(idx) + ["cdd"] * len(idx),
                "metric_value": np.concatenate((hdd, cdd)),
            })
        pivot = df.pivot_table(index="date", columns="metric_name", values="metric_value", aggfunc="last")
        pivot = pivot.sort_index()
        return pivot

    def _prepare_design(self, prices: pd.Series, hdd: pd.Series, cdd: pd.Series) -> np.ndarray:
        aligned = pd.concat([prices, hdd, cdd], axis=1, join="inner").dropna()
        aligned = aligned.tail(self.window)
        aligned.columns = ["price", "hdd", "cdd"]
        if len(aligned) < 30:
            raise ValueError("Insufficient observations for weather regression")
        self._aligned = aligned
        intercept = np.ones(len(aligned))
        dow = pd.get_dummies(aligned.index.to_series().apply(lambda d: d.weekday()), prefix="dow")
        X = np.column_stack([intercept, aligned["hdd"].values, aligned["cdd"].values])
        if not dow.empty:
            X = np.column_stack([X, dow.values])
        self._feature_names = ["intercept", "hdd", "cdd"] + list(dow.columns)
        self._response = aligned["price"].values
        return X

    def run(self, entity_id: str, as_of: date) -> List[WeatherImpactCoefficient]:
        prices_df = self._load_prices(entity_id, as_of)
        dd_df = self._load_degree_days(entity_id, as_of)
        prices_series = prices_df.set_index("date")["price"]
        hdd_series = dd_df.get("hdd", pd.Series(dtype=float))
        cdd_series = dd_df.get("cdd", pd.Series(dtype=float))

        X = self._prepare_design(prices_series, hdd_series, cdd_series)
        y = self._response

        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        resid = y - y_pred
        dof = len(y) - len(beta)
        sigma2 = float(resid @ resid / max(dof, 1))
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        std_err = np.sqrt(np.diag(cov))

        ss_total = float(((y - y.mean()) ** 2).sum())
        ss_res = float((resid ** 2).sum())
        r2 = 1 - ss_res / ss_total if ss_total else 0.0

        diagnostics = {
            "residual_std": float(np.sqrt(sigma2)),
            "n_obs": len(y),
            "window_days": self.window,
        }

        coeffs = []
        for idx, name in enumerate(self._feature_names):
            if name not in {"hdd", "cdd"}:
                continue
            coeffs.append(
                WeatherImpactCoefficient(
                    as_of_date=as_of,
                    entity_id=entity_id,
                    coef_type=name,
                    coefficient=float(beta[idx] if idx < len(beta) else 0.0),
                    r2=r2,
                    window=f"{self.window}d",
                    diagnostics={
                        **diagnostics,
                        "std_error": float(std_err[idx] if idx < len(std_err) else 0.0),
                        "last_price": float(self._aligned["price"].iloc[-1]),
                    },
                )
            )
        return coeffs

    def persist(self, coeffs: List[WeatherImpactCoefficient]) -> None:
        if not coeffs:
            return
        rows = [{
            "date": c.as_of_date,
            "entity_id": c.entity_id,
            "coef_type": c.coef_type,
            "coefficient": c.coefficient,
            "r2": c.r2,
            "p_value": None,
            "window": c.window,
            "extreme_event_count": 0,
            "diagnostics": json.dumps(c.diagnostics),
            "method": "ols",
            "model_version": "v1",
        } for c in coeffs]
        insert_rows("market_intelligence.weather_impact", rows)
