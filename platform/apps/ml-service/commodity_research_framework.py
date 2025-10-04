"""Commodity Research Framework.

Provides advanced analytics for energy commodities including decomposition,
volatility regime detection, supply-demand balance modeling, and weather
impact calibration. Persistence hooks are exposed for the wider platform.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    KMeans = None  # type: ignore
    StandardScaler = None  # type: ignore

from data_access import DataAccessLayer

try:  # hmmlearn is optional in some environments
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None  # type: ignore

try:  # statsmodels is required for advanced analytics but guarded for safety
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tsa.seasonal import STL  # type: ignore
    from statsmodels.tsa.regime_switching.markov_regression import (  # type: ignore
        MarkovRegression,
    )
except Exception:  # pragma: no cover - optional dependency guard
    sm = None  # type: ignore
    STL = None  # type: ignore
    MarkovRegression = None  # type: ignore


logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> Optional[float]:
    """Convert numeric-like values to finite floats suitable for persistence."""
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


@dataclass
class DecompositionResult:
    instrument_id: str
    commodity_type: str
    method: str
    components: Dict[str, pd.Series]
    snapshot_date: datetime
    version: str = "v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_persist_rows(self) -> List[Tuple[Any, ...]]:
        trend = self.components.get("trend", pd.Series(dtype=float))
        seasonal = self.components.get("seasonal", pd.Series(dtype=float))
        residual = self.components.get("residual", pd.Series(dtype=float))
        df = pd.concat(
            [
                trend.rename("trend"),
                seasonal.rename("seasonal"),
                residual.rename("residual"),
            ],
            axis=1,
        ).sort_index()

        rows: List[Tuple[Any, ...]] = []
        for ts, row in df.iterrows():
            component_date = pd.to_datetime(ts).date()
            rows.append(
                (
                    component_date,
                    self.instrument_id,
                    self.method,
                    _safe_float(row.get("trend")),
                    _safe_float(row.get("seasonal")),
                    _safe_float(row.get("residual")),
                    self.version,
                    self.snapshot_date,
                )
            )
        return rows


@dataclass
class VolatilityRegimeResult:
    instrument_id: str
    method: str
    labels: pd.Series
    features: pd.DataFrame
    regime_profiles: Dict[str, Dict[str, float]]
    n_regimes: int
    fit_version: str = "v1"
    metadata: Dict[str, Any] = field(default_factory=dict)
    as_of: datetime = field(default_factory=datetime.utcnow)

    def to_persist_rows(self) -> List[Tuple[Any, ...]]:
        aligned = self.features.join(self.labels.rename("regime_label"), how="inner")
        rows: List[Tuple[Any, ...]] = []
        for ts, row in aligned.iterrows():
            row_date = pd.to_datetime(ts).date()
            feature_payload = {
                key: _safe_float(row.get(key))
                for key in self.features.columns
            }
            rows.append(
                (
                    row_date,
                    self.instrument_id,
                    str(row.get("regime_label")),
                    json.dumps(feature_payload),
                    self.method,
                    self.n_regimes,
                    self.fit_version,
                    self.as_of,
                )
            )
        return rows


@dataclass
class SupplyDemandResult:
    entity_id: str
    instrument_id: Optional[str]
    metrics: Dict[str, pd.Series]
    units: Dict[str, str]
    as_of: datetime
    version: str = "v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_persist_rows(self) -> List[Tuple[Any, ...]]:
        rows: List[Tuple[Any, ...]] = []
        for metric_name, series in self.metrics.items():
            if series is None or series.empty:
                continue
            for ts, value in series.sort_index().items():
                metric_date = pd.to_datetime(ts).date()
                rows.append(
                    (
                        metric_date,
                        self.entity_id or self.instrument_id or "unknown",
                        self.instrument_id,
                        metric_name,
                        _safe_float(value),
                        self.units.get(metric_name, ""),
                        self.version,
                        self.as_of,
                    )
                )
        return rows


@dataclass
class WeatherImpactResult:
    entity_id: str
    coefficients: Dict[str, Dict[str, float]]
    r_squared: Optional[float]
    window: str
    model_version: str
    as_of: datetime
    method: str = "ols"
    extreme_event_count: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_persist_rows(self) -> List[Tuple[Any, ...]]:
        rows: List[Tuple[Any, ...]] = []
        for coef_type, stats_dict in self.coefficients.items():
            diag_payload = self.diagnostics.get(coef_type)
            rows.append(
                (
                    self.as_of.date(),
                    self.entity_id,
                    coef_type,
                    _safe_float(stats_dict.get("coef")),
                    _safe_float(self.r_squared),
                    _safe_float(stats_dict.get("p_value")),
                    self.window,
                    self.model_version,
                    self.extreme_event_count,
                    json.dumps(diag_payload) if diag_payload is not None else None,
                    self.method,
                    self.as_of,
                )
            )
        return rows


class CommodityResearchFramework:
    """Comprehensive research framework for energy commodity analysis."""

    def __init__(
        self,
        *,
        data_access: Optional[DataAccessLayer] = None,
        persistence: Optional[Any] = None,
    ) -> None:
        self.data_access = data_access or DataAccessLayer()
        self.persistence = persistence
        self.volatility_regimes: Dict[str, VolatilityRegimeResult] = {}
        self.seasonal_patterns: Dict[str, DecompositionResult] = {}
        self.weather_sensitivities: Dict[str, WeatherImpactResult] = {}

    # ------------------------------------------------------------------
    # Persistence configuration
    # ------------------------------------------------------------------
    def set_persistence(self, persistence: Any) -> None:
        """Inject persistence layer after initialization."""
        self.persistence = persistence

    # ------------------------------------------------------------------
    # Time series decomposition
    # ------------------------------------------------------------------
    def generate_time_series_decomposition(
        self,
        instrument_id: str,
        commodity_type: str,
        *,
        method: str = "stl",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        version: str = "v1",
        persist: bool = False,
    ) -> DecompositionResult:
        """Compute decomposition for an instrument and optionally persist it."""
        prices = self.data_access.get_price_series(
            instrument_id,
            start=start,
            end=end,
            lookback_days=365,
        )
        if prices.empty:
            raise ValueError(f"No price data for instrument {instrument_id}")

        components = self.decompose_time_series(
            prices=prices,
            commodity_type=commodity_type,
            decomposition_method=method,
        )
        result = DecompositionResult(
            instrument_id=instrument_id,
            commodity_type=commodity_type,
            method=method,
            components=components,
            snapshot_date=datetime.utcnow(),
            version=version,
            metadata={
                "sample_size": int(len(prices)),
                "start": prices.index.min().isoformat(),
                "end": prices.index.max().isoformat(),
                "seasonal_period": self._determine_seasonal_period(commodity_type),
            },
        )

        self.seasonal_patterns[instrument_id] = result
        if persist and self.persistence:
            try:
                self.persistence.persist_decomposition(result)
            except Exception as exc:  # pragma: no cover - external IO guard
                logger.exception("Failed to persist decomposition: %s", exc)

        return result

    def decompose_time_series(
        self,
        prices: pd.Series,
        commodity_type: str,
        decomposition_method: str = "stl",
    ) -> Dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, residual components."""
        prices = prices.sort_index().dropna()
        if prices.empty:
            raise ValueError("Price series is empty for decomposition")

        logger.info(
            "Decomposing %s time series using %s",
            commodity_type,
            decomposition_method,
        )

        if decomposition_method == "stl":
            return self._stl_decomposition(prices, commodity_type)
        if decomposition_method == "classical":
            return self._classical_decomposition(prices, commodity_type)
        raise ValueError(f"Unknown decomposition method: {decomposition_method}")

    def _determine_seasonal_period(self, commodity_type: str) -> int:
        mapping = {
            "gas": 7,
            "power": 7,
            "oil": 365,
        }
        return mapping.get(commodity_type.lower(), 30)

    def _stl_decomposition(
        self,
        prices: pd.Series,
        commodity_type: str,
    ) -> Dict[str, pd.Series]:
        if STL is None:
            logger.warning(
                "statsmodels STL unavailable; falling back to classical decomposition"
            )
            return self._classical_decomposition(prices, commodity_type)

        seasonal_period = self._determine_seasonal_period(commodity_type)
        stl = STL(prices, period=seasonal_period, robust=True)
        result = stl.fit()
        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        }

    def _classical_decomposition(
        self,
        prices: pd.Series,
        commodity_type: str,
    ) -> Dict[str, pd.Series]:
        if commodity_type.lower() == "power":
            trend_window = 30
        else:
            trend_window = 90

        trend = prices.rolling(trend_window, center=True, min_periods=max(3, trend_window // 2)).mean()
        detrended = prices - trend

        if len(prices) > self._determine_seasonal_period(commodity_type) * 2:
            seasonal = self._extract_seasonal_component(detrended, commodity_type)
        else:
            seasonal = pd.Series(0.0, index=prices.index)

        residual = prices - trend - seasonal
        trend = trend.bfill().ffill()

        return {
            "trend": trend,
            "seasonal": seasonal.fillna(0.0),
            "residual": residual.fillna(0.0),
        }

    def _extract_seasonal_component(
        self,
        detrended: pd.Series,
        commodity_type: str,
    ) -> pd.Series:
        detrended = detrended.dropna()
        if detrended.empty:
            return pd.Series(0.0, index=detrended.index)

        if commodity_type.lower() in {"power"}:
            grouped = detrended.groupby(detrended.index.dayofweek).mean()
            seasonal = detrended.index.dayofweek.map(grouped)
        else:
            grouped = detrended.groupby(detrended.index.dayofyear).mean()
            seasonal = detrended.index.dayofyear.map(grouped)

        return pd.Series(seasonal.values, index=detrended.index).fillna(0.0)

    # ------------------------------------------------------------------
    # Volatility regimes
    # ------------------------------------------------------------------
    def analyze_volatility_regimes(
        self,
        instrument_id: str,
        *,
        n_regimes: int = 3,
        method: str = "auto",
        lookback_days: int = 365,
        version: str = "v1",
        persist: bool = False,
    ) -> VolatilityRegimeResult:
        """Fetch prices, compute returns, detect regimes, and persist if requested."""
        returns = self.data_access.get_return_series(
            instrument_id,
            lookback_days=lookback_days,
        )
        if returns.empty:
            raise ValueError(f"Insufficient returns data for {instrument_id}")

        result = self.detect_volatility_regimes(
            returns=returns,
            n_regimes=n_regimes,
            method=method,
            instrument_id=instrument_id,
            version=version,
        )

        if persist and self.persistence:
            try:
                self.persistence.persist_volatility_regimes(result)
            except Exception as exc:  # pragma: no cover - external IO guard
                logger.exception("Failed to persist volatility regimes: %s", exc)

        return result

    def detect_volatility_regimes(
        self,
        returns: pd.Series,
        *,
        n_regimes: int = 3,
        method: str = "auto",
        instrument_id: Optional[str] = None,
        version: str = "v1",
    ) -> VolatilityRegimeResult:
        """Detect volatility regimes using clustering, Markov switching, or HMM."""
        returns = returns.sort_index().dropna()
        features = self._build_volatility_features(returns)
        if features.empty:
            raise ValueError("Insufficient data after feature engineering for regimes")

        methods: Sequence[str]
        if method == "auto":
            methods = ("kmeans", "markov", "hmm")
        else:
            methods = (method,)

        best_result: Optional[VolatilityRegimeResult] = None
        best_bic = np.inf

        for candidate in methods:
            try:
                candidate_result = self._fit_regime_model(
                    candidate,
                    returns,
                    features,
                    n_regimes=n_regimes,
                    instrument_id=instrument_id,
                    version=version,
                )
            except Exception as exc:
                logger.warning("Regime model %s failed: %s", candidate, exc)
                continue

            bic = candidate_result.metadata.get("bic", np.inf)
            if bic is None:
                bic = np.inf
            if bic < best_bic:
                best_bic = bic
                best_result = candidate_result

        if best_result is None:
            raise RuntimeError("All volatility regime methods failed")

        key = instrument_id or "returns"
        self.volatility_regimes[key] = best_result
        return best_result

    def _build_volatility_features(self, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=returns.index)
        df["volatility_20d"] = returns.rolling(20).std()
        df["volatility_60d"] = returns.rolling(60).std()
        df["skew_60d"] = returns.rolling(60).skew()
        df["kurtosis_60d"] = returns.rolling(60).kurt()

        def _autocorr(x: pd.Series) -> float:
            return x.autocorr(lag=1) if len(x) > 1 else np.nan

        df["autocorr_20d"] = returns.rolling(20).apply(_autocorr, raw=False)
        df["realized_vol_5d"] = returns.rolling(5).std() * np.sqrt(252)
        df["realized_vol_20d"] = returns.rolling(20).std() * np.sqrt(252)
        df["realized_vol_60d"] = returns.rolling(60).std() * np.sqrt(252)
        threshold = returns.std() * 3 if returns.std() > 0 else 0
        df["jump_indicator"] = (returns.abs() > threshold).astype(int)

        return df.dropna()

    def _fit_regime_model(
        self,
        method: str,
        returns: pd.Series,
        features: pd.DataFrame,
        *,
        n_regimes: int,
        instrument_id: Optional[str],
        version: str,
    ) -> VolatilityRegimeResult:
        if method == "kmeans":
            if KMeans is None or StandardScaler is None:
                raise RuntimeError("scikit-learn is required for KMeans regime detection")
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)
            model = KMeans(n_clusters=n_regimes, random_state=42)
            labels = model.fit_predict(scaled)
            inertia = float(model.inertia_)
            n_samples, n_features = scaled.shape
            bic = inertia + 0.5 * n_regimes * n_features * np.log(n_samples)
            label_series = pd.Series(labels, index=features.index, name="regime_label")
            metadata = {"bic": bic, "inertia": inertia}
        elif method == "markov":
            if MarkovRegression is None or sm is None:
                raise RuntimeError("statsmodels is not available for Markov switching")
            aligned_returns = returns.reindex(features.index).dropna()
            if aligned_returns.empty:
                raise ValueError("Returns alignment failed for Markov model")
            model = MarkovRegression(
                aligned_returns,
                k_regimes=n_regimes,
                trend="c",
                switching_variance=True,
            )
            fit_result = model.fit(disp=False)
            probs = fit_result.smoothed_marginal_probabilities.T
            labels = probs.idxmax(axis=0).astype(int)
            label_series = pd.Series(labels.values, index=probs.columns, name="regime_label")
            label_series = label_series.sort_index()
            label_series = label_series.reindex(features.index, method="ffill").dropna()
            features = features.loc[label_series.index]
            bic = float(fit_result.bic)
            metadata = {
                "bic": bic,
                "loglike": float(fit_result.llf),
            }
        elif method == "hmm":
            if GaussianHMM is None:
                raise RuntimeError("hmmlearn is not available for HMM regimes")
            aligned_returns = returns.reindex(features.index).dropna()
            if aligned_returns.empty:
                raise ValueError("Returns alignment failed for HMM")
            model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=500)
            X = aligned_returns.values.reshape(-1, 1)
            model.fit(X)
            states = model.predict(X)
            label_series = pd.Series(states, index=aligned_returns.index, name="regime_label")
            features = features.loc[label_series.index]
            log_prob = model.score(X)
            n_params = n_regimes * 3  # mean, variance, transition approx
            bic = -2 * log_prob + n_params * np.log(len(X))
            metadata = {
                "bic": float(bic),
                "loglike": float(log_prob),
            }
        else:
            raise ValueError(f"Unsupported regime method: {method}")

        regime_profiles = {
            f"regime_{int(label)}": {
                feature: _safe_float(values)
                for feature, values in group.mean().items()
            }
            for label, group in features.groupby(label_series)
        }

        result = VolatilityRegimeResult(
            instrument_id=instrument_id or "returns",
            method=method,
            labels=label_series.astype(str),
            features=features,
            regime_profiles=regime_profiles,
            n_regimes=n_regimes,
            fit_version=version,
            metadata=metadata,
            as_of=datetime.utcnow(),
        )
        return result

    # ------------------------------------------------------------------
    # Supply / demand balance
    # ------------------------------------------------------------------
    def model_supply_demand_balance(
        self,
        prices: pd.Series,
        inventory_data: Optional[pd.Series] = None,
        production_data: Optional[pd.Series] = None,
        consumption_data: Optional[pd.Series] = None,
        *,
        instrument_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        version: str = "v1",
        persist: bool = False,
    ) -> SupplyDemandResult:
        """Model supply-demand equilibrium metrics for a commodity."""
        prices = prices.sort_index().dropna()
        metrics: Dict[str, pd.Series] = {}
        units: Dict[str, str] = {}

        inventory_unit = None
        if inventory_data is not None and hasattr(inventory_data, "attrs"):
            inventory_unit = inventory_data.attrs.get("unit")

        production_unit = None
        if production_data is not None and hasattr(production_data, "attrs"):
            production_unit = production_data.attrs.get("unit")

        inventory_series = (
            inventory_data.rename("inventory")
            if inventory_data is not None
            else pd.Series(dtype=float, name="inventory")
        )
        production_series = (
            production_data.rename("production")
            if production_data is not None
            else pd.Series(dtype=float, name="production")
        )
        consumption_series = (
            consumption_data.rename("consumption")
            if consumption_data is not None
            else pd.Series(dtype=float, name="consumption")
        )

        joined = pd.concat(
            [
                prices.rename("price"),
                inventory_series,
                production_series,
                consumption_series,
            ],
            axis=1,
        ).dropna(how="all")

        if "inventory" in joined and "consumption" in joined:
            inv = joined["inventory"].interpolate(limit_direction="both")
            cons = joined["consumption"].replace(0, np.nan).interpolate(limit_direction="both")
            cover_days = (inv / cons) * 30
            metrics["inventory_cover_days"] = cover_days
            units["inventory_cover_days"] = "days"

            monthly_build = inv.resample("M").last().diff().reindex(joined.index, method="ffill")
            metrics["inventory_monthly_build"] = monthly_build
            units["inventory_monthly_build"] = inventory_unit or "units"

        if "production" in joined and "consumption" in joined:
            balance = joined["production"].ffill() - joined["consumption"].ffill()
            metrics["supply_demand_balance"] = balance
            units["supply_demand_balance"] = production_unit or "units"

            cons = joined["consumption"].replace(0, np.nan)
            balance_pct = (balance / cons) * 100
            metrics["balance_pct_consumption"] = balance_pct
            units["balance_pct_consumption"] = "percent"

            cusum_flags = self._detect_cusum_shocks(balance)
            metrics["cusum_supply_shock"] = cusum_flags
            units["cusum_supply_shock"] = "flag"

        if "inventory" in joined:
            inv_changes = joined["inventory"].diff()
            price_changes = joined["price"].diff()
            corr = price_changes.rolling(90).corr(inv_changes)
            metrics["price_inventory_correlation"] = corr
            units["price_inventory_correlation"] = "correlation"

        result = SupplyDemandResult(
            entity_id=entity_id or (instrument_id or "unknown"),
            instrument_id=instrument_id,
            metrics=metrics,
            units=units,
            as_of=datetime.utcnow(),
            version=version,
            metadata={
                "price_start": prices.index.min().isoformat() if not prices.empty else None,
                "price_end": prices.index.max().isoformat() if not prices.empty else None,
            },
        )

        if persist and self.persistence:
            try:
                self.persistence.persist_supply_demand(result)
            except Exception as exc:  # pragma: no cover - external IO guard
                logger.exception("Failed to persist supply/demand metrics: %s", exc)

        return result

    def _detect_cusum_shocks(self, series: pd.Series, threshold_factor: float = 1.5) -> pd.Series:
        series = series.ffill().dropna()
        if series.empty:
            return pd.Series(dtype=float)

        mean = series.mean()
        std = series.std() or 1.0
        threshold = std * threshold_factor
        pos_cusum = np.zeros(len(series))
        neg_cusum = np.zeros(len(series))

        for idx, value in enumerate(series - mean):
            pos_cusum[idx] = max(0, pos_cusum[idx - 1] + value - threshold)
            neg_cusum[idx] = min(0, neg_cusum[idx - 1] + value + threshold)

        flags = (pos_cusum > 0) | (neg_cusum < 0)
        return pd.Series(flags.astype(int), index=series.index)

    # ------------------------------------------------------------------
    # Weather impact analysis
    # ------------------------------------------------------------------
    def analyze_weather_impact(
        self,
        prices: pd.Series,
        temperature_data: pd.Series,
        heating_demand: Optional[pd.Series] = None,
        cooling_demand: Optional[pd.Series] = None,
        *,
        entity_id: Optional[str] = None,
        window: str = "90D",
        lags: Optional[Sequence[int]] = None,
        quantile_levels: Optional[Sequence[float]] = None,
        model_version: str = "v1",
        persist: bool = False,
    ) -> WeatherImpactResult:
        """Calibrate weather sensitivities using OLS with optional quantiles."""
        prices = prices.sort_index().astype(float)
        base_frame = pd.DataFrame(
            {
                "price": prices,
                "temperature": temperature_data,
                "hdd": (heating_demand if heating_demand is not None else pd.Series(dtype=float)),
                "cdd": (cooling_demand if cooling_demand is not None else pd.Series(dtype=float)),
            }
        )
        df = base_frame.dropna(subset=["price", "temperature"], how="any").sort_index()
        df.index = pd.to_datetime(df.index)
        if df.empty:
            raise ValueError("Insufficient overlapping data for weather impact analysis")

        if df["temperature"].std(skipna=True) == 0:
            raise ValueError("Temperature series has no variance for weather impact analysis")

        if lags is None:
            lags = (1, 3, 7)
        for lag in lags:
            df[f"temperature_lag_{lag}"] = df["temperature"].shift(lag)

        df["month"] = df.index.month
        seasonal_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
        model_df = pd.concat([df, seasonal_dummies], axis=1).dropna()

        feature_cols = [col for col in model_df.columns if col != "price"]
        if sm is None:
            raise RuntimeError("statsmodels is required for weather impact analysis")

        X = sm.add_constant(model_df[feature_cols])
        y = model_df["price"]
        ols_model = sm.OLS(y, X, missing="drop")
        ols_result = ols_model.fit()

        coefficients: Dict[str, Dict[str, float]] = {}
        diagnostics: Dict[str, Any] = {}
        for column in feature_cols:
            coefficients[column] = {
                "coef": ols_result.params.get(column, np.nan),
                "p_value": ols_result.pvalues.get(column, np.nan),
            }

        if quantile_levels is None:
            quantile_levels = (0.1, 0.9)
        quantile_stats: Dict[str, Dict[str, float]] = {}
        try:
            quant_model = sm.QuantReg(y, X)
            for alpha in quantile_levels:
                quant_fit = quant_model.fit(q=alpha)
                quantile_stats[f"quantile_{alpha}"] = {
                    feature: quant_fit.params.get(feature, np.nan)
                    for feature in feature_cols
                }
        except Exception as exc:  # pragma: no cover - optional path
            logger.warning("Quantile regression failed: %s", exc)

        # Rolling stability diagnostics
        window_td = pd.Timedelta(window)
        window_days = max(int(window_td.days or window_td.components.days or 0), 30)
        rolling_window = window_days
        rolling_diag: Dict[str, Any] = {}
        for feature in ["temperature", "hdd", "cdd"]:
            if feature in model_df:
                series = model_df[feature]
                cov = series.rolling(rolling_window).cov(model_df["price"])
                var = series.rolling(rolling_window).var()
                rolling_beta = cov / var
                rolling_diag[feature] = {
                    "avg_beta": _safe_float(rolling_beta.mean()),
                    "beta_std": _safe_float(rolling_beta.std()),
                }

        temp_std = temperature_data.std()
        extreme_temp = temperature_data[
            (temperature_data > temperature_data.mean() + 2 * temp_std)
            | (temperature_data < temperature_data.mean() - 2 * temp_std)
        ]
        extreme_count = int(len(extreme_temp))
        if heating_demand is not None:
            extreme_count += int((heating_demand > heating_demand.mean() + 2 * heating_demand.std()).sum())
        if cooling_demand is not None:
            extreme_count += int((cooling_demand > cooling_demand.mean() + 2 * cooling_demand.std()).sum())

        diagnostics.update({f"quantile_{k}": v for k, v in quantile_stats.items()})
        diagnostics.update(rolling_diag)

        result = WeatherImpactResult(
            entity_id=entity_id or "unknown",
            coefficients=coefficients,
            r_squared=_safe_float(ols_result.rsquared),
            window=window,
            model_version=model_version,
            as_of=datetime.utcnow(),
            method="ols",
            extreme_event_count=extreme_count,
            diagnostics=diagnostics,
        )

        self.weather_sensitivities[result.entity_id] = result
        if persist and self.persistence:
            try:
                self.persistence.persist_weather_impact(result)
            except Exception as exc:  # pragma: no cover - external IO guard
                logger.exception("Failed to persist weather impact: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Portfolio metrics (retained from previous implementation)
    # ------------------------------------------------------------------
    def calculate_fundamental_metrics(
        self,
        prices: pd.Series,
        commodity_type: str,
    ) -> Dict[str, float]:
        prices = prices.sort_index().dropna()
        if prices.empty:
            return {"error": "Insufficient data for metrics calculation"}

        returns = prices.pct_change().dropna()
        if len(returns) < 30:
            return {"error": "Insufficient data for metrics calculation"}

        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else float("inf")

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0.0
        sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0

        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

        return {
            "mean_annual_return": float(mean_return),
            "annual_volatility": float(std_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "sortino_ratio": float(sortino_ratio),
            "var_95_1d": float(var_95),
            "cvar_95_1d": float(cvar_95),
            "commodity_type": commodity_type,
        }
