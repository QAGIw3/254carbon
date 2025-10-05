"""Commodity feature engineering utilities shared across services."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class CommodityFeatureEngineer:
    """Unified commodity feature engineering for analytics and ML domains."""

    def build_generation_spread_features(
        self,
        *,
        power_prices: pd.Series,
        gas_prices: Optional[pd.Series] = None,
        coal_prices: Optional[pd.Series] = None,
        carbon_prices: Optional[pd.Series] = None,
        heat_rate_gas: Optional[float] = None,
        heat_rate_coal: Optional[float] = None,
        emissions_factor_gas: Optional[float] = None,
        emissions_factor_coal: Optional[float] = None,
        fallback_capacity: Optional[float] = None,
        capacity_series: Optional[pd.Series] = None,
        load_series: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Compute spark/dark spreads and capacity utilization metrics."""

        if power_prices is None or power_prices.empty:
            return {}

        power = power_prices.sort_index()

        def _latest(series: pd.Series) -> float:
            valid = series.dropna()
            return float(valid.iloc[-1]) if not valid.empty else 0.0

        def _tail_mean(series: pd.Series, window: int = 30) -> float:
            valid = series.dropna()
            if valid.empty:
                return 0.0
            window = min(window, len(valid))
            return float(valid.iloc[-window:].mean())

        def _tail_std(series: pd.Series, window: int = 30) -> float:
            valid = series.dropna()
            if valid.empty:
                return 0.0
            window = min(window, len(valid))
            return float(valid.iloc[-window:].std())

        features: Dict[str, float] = {}

        if gas_prices is not None and not gas_prices.empty and heat_rate_gas:
            spark_df = pd.concat(
                {"power": power, "gas": gas_prices.sort_index()}, axis=1
            ).dropna()
            if not spark_df.empty:
                if carbon_prices is not None and not carbon_prices.empty:
                    spark_df = spark_df.join(
                        carbon_prices.sort_index().rename("carbon"), how="left"
                    )
                    spark_df["carbon"].fillna(method="ffill", inplace=True)
                    spark_df["carbon"].fillna(method="bfill", inplace=True)
                    spark_df["carbon"].fillna(0.0, inplace=True)
                else:
                    spark_df["carbon"] = 0.0

                spark_df["spark_spread"] = spark_df["power"] - heat_rate_gas * spark_df["gas"]
                spark_df["clean_spark_spread"] = spark_df["spark_spread"] - (
                    (emissions_factor_gas or 0.0) * spark_df["carbon"]
                )
                spark_df["spark_margin_ratio"] = np.where(
                    spark_df["power"] != 0,
                    spark_df["spark_spread"] / spark_df["power"],
                    0.0,
                )

                features.update(
                    {
                        "spark_spread_latest": _latest(spark_df["spark_spread"]),
                        "spark_spread_30d_mean": _tail_mean(spark_df["spark_spread"]),
                        "spark_spread_30d_std": _tail_std(spark_df["spark_spread"]),
                        "clean_spark_spread_latest": _latest(
                            spark_df["clean_spark_spread"]
                        ),
                        "clean_spark_spread_30d_mean": _tail_mean(
                            spark_df["clean_spark_spread"]
                        ),
                        "spark_margin_ratio_latest": _latest(
                            spark_df["spark_margin_ratio"]
                        ),
                    }
                )

        if coal_prices is not None and not coal_prices.empty and heat_rate_coal:
            dark_df = pd.concat(
                {"power": power, "coal": coal_prices.sort_index()}, axis=1
            ).dropna()
            if not dark_df.empty:
                if carbon_prices is not None and not carbon_prices.empty:
                    dark_df = dark_df.join(
                        carbon_prices.sort_index().rename("carbon"), how="left"
                    )
                    dark_df["carbon"].fillna(method="ffill", inplace=True)
                    dark_df["carbon"].fillna(method="bfill", inplace=True)
                    dark_df["carbon"].fillna(0.0, inplace=True)
                else:
                    dark_df["carbon"] = 0.0

                dark_df["dark_spread"] = dark_df["power"] - heat_rate_coal * dark_df["coal"]
                dark_df["clean_dark_spread"] = dark_df["dark_spread"] - (
                    (emissions_factor_coal or 0.0) * dark_df["carbon"]
                )

                features.update(
                    {
                        "dark_spread_latest": _latest(dark_df["dark_spread"]),
                        "dark_spread_30d_mean": _tail_mean(dark_df["dark_spread"]),
                        "dark_spread_30d_std": _tail_std(dark_df["dark_spread"]),
                        "clean_dark_spread_latest": _latest(
                            dark_df["clean_dark_spread"]
                        ),
                        "clean_dark_spread_30d_mean": _tail_mean(
                            dark_df["clean_dark_spread"]
                        ),
                    }
                )

        if load_series is not None and not load_series.empty:
            load = load_series.sort_index()
            if capacity_series is not None and not capacity_series.empty:
                capacity = capacity_series.sort_index().reindex(load.index)
                capacity = capacity.fillna(method="ffill").fillna(method="bfill")
            elif fallback_capacity is not None:
                capacity = pd.Series(fallback_capacity, index=load.index)
            else:
                capacity = None

            if capacity is not None:
                capacity_df = pd.concat(
                    {"load": load, "capacity": capacity}, axis=1
                ).dropna()
                if not capacity_df.empty:
                    capacity_df["capacity_utilization"] = np.where(
                        capacity_df["capacity"] != 0,
                        capacity_df["load"] / capacity_df["capacity"],
                        0.0,
                    )
                    capacity_df["reserve_margin"] = np.where(
                        capacity_df["capacity"] != 0,
                        (capacity_df["capacity"] - capacity_df["load"]) / capacity_df["capacity"],
                        0.0,
                    )
                    capacity_df["capacity_headroom"] = (
                        capacity_df["capacity"] - capacity_df["load"]
                    )

                    features.update(
                        {
                            "capacity_utilization_latest": _latest(
                                capacity_df["capacity_utilization"]
                            ),
                            "capacity_utilization_30d_mean": _tail_mean(
                                capacity_df["capacity_utilization"]
                            ),
                            "reserve_margin_latest": _latest(
                                capacity_df["reserve_margin"]
                            ),
                            "reserve_margin_30d_mean": _tail_mean(
                                capacity_df["reserve_margin"]
                            ),
                            "capacity_headroom_latest": _latest(
                                capacity_df["capacity_headroom"]
                            ),
                        }
                    )

        return features

