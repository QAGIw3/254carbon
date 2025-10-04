"""
Multi-Commodity Feature Engineering

Generates sophisticated features across energy commodities for ML forecasting:
- Cross-commodity correlations and spreads
- Crack spreads and refining economics
- Seasonality patterns by commodity type
- Storage and inventory dynamics
- Weather and demand correlations
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class CommodityFeatureEngineer:
    """
    Advanced feature engineering for multi-commodity energy markets.

    Features:
    - Cross-commodity correlations and cointegration
    - Crack spreads and refining economics
    - Seasonal patterns and calendar effects
    - Storage arbitrage opportunities
    - Weather-demand relationships
    """

    def __init__(self):
        self.commodity_groups = {
            "oil_complex": ["WTI", "BRENT", "DUBAI", "WTI_CUSHING"],
            "gas_complex": ["HENRY_HUB", "TTF", "NBP", "JKM"],
            "coal_complex": ["API2", "API4", "NEWCASTLE", "RICHARDS_BAY"],
            "refined_products": ["GASOLINE", "DIESEL", "JET_FUEL", "HEATING_OIL"],
            "emissions": ["EU_ETS", "RGGI", "CCA", "VOLUNTARY"]
        }

    def build_cross_commodity_features(
        self,
        price_data: pd.DataFrame,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Build features based on relationships between different commodities.

        Features:
        - Correlation coefficients
        - Spread ratios
        - Cointegration statistics
        - Relative strength indices
        """
        features = []

        for group_name, commodities in self.commodity_groups.items():
            group_prices = price_data[commodities].dropna()

            if len(group_prices.columns) < 2:
                continue

            # Rolling correlations
            for i, comm1 in enumerate(commodities[:-1]):
                for comm2 in commodities[i+1:]:
                    if comm1 in group_prices.columns and comm2 in group_prices.columns:
                        corr_series = group_prices[comm1].rolling(lookback_days).corr(
                            group_prices[comm2]
                        )

                        features.append({
                            'feature_name': f'corr_{comm1}_{comm2}_{lookback_days}d',
                            'values': corr_series.fillna(0)
                        })

            # Spread ratios (e.g., Brent/WTI)
            if 'BRENT' in group_prices.columns and 'WTI' in group_prices.columns:
                brent_wti_spread = (group_prices['BRENT'] / group_prices['WTI'] - 1) * 100
                features.append({
                    'feature_name': 'brent_wti_spread_pct',
                    'values': brent_wti_spread
                })

        return self._combine_features(features)

    def build_crack_spread_features(
        self,
        crude_prices: pd.Series,
        product_prices: pd.Series,
        crack_type: str = "3:2:1"
    ) -> pd.DataFrame:
        """
        Calculate crack spread features for refining economics.

        Args:
            crude_prices: WTI or Brent prices
            product_prices: Dict of gasoline, diesel, heating oil prices
            crack_type: "3:2:1" (3 crude : 2 gasoline : 1 diesel)
                       "5:3:2" (5 crude : 3 gasoline : 2 diesel)
        """
        features = []

        if crack_type == "3:2:1":
            # 3 barrels crude -> 2 barrels gasoline + 1 barrel diesel
            if 'GASOLINE' in product_prices.columns and 'DIESEL' in product_prices.columns:
                crack_value = (
                    2 * product_prices['GASOLINE'] +
                    1 * product_prices['DIESEL'] -
                    3 * crude_prices
                )
                features.append({
                    'feature_name': 'crack_spread_3_2_1',
                    'values': crack_value
                })

        elif crack_type == "5:3:2":
            # 5 barrels crude -> 3 barrels gasoline + 2 barrels diesel
            if 'GASOLINE' in product_prices.columns and 'DIESEL' in product_prices.columns:
                crack_value = (
                    3 * product_prices['GASOLINE'] +
                    2 * product_prices['DIESEL'] -
                    5 * crude_prices
                )
                features.append({
                    'feature_name': 'crack_spread_5_3_2',
                    'values': crack_value
                })

        # Crack spread volatility
        if features:
            crack_series = features[0]['values']
            crack_vol = crack_series.rolling(30).std()
            features.append({
                'feature_name': f'crack_volatility_{crack_type}',
                'values': crack_vol.fillna(0)
            })

        return self._combine_features(features)

    def build_seasonality_features(
        self,
        price_series: pd.Series,
        commodity_type: str
    ) -> pd.DataFrame:
        """
        Build seasonal and calendar-based features.

        Features:
        - Monthly seasonality
        - Quarterly patterns
        - Holiday effects
        - Weather-related seasonality
        """
        features = []

        # Monthly seasonality (detrended)
        monthly_avg = price_series.groupby(price_series.index.month).mean()
        monthly_seasonal = price_series.index.month.map(monthly_avg)
        features.append({
            'feature_name': f'{commodity_type}_monthly_seasonal',
            'values': (monthly_seasonal - monthly_seasonal.mean()) / monthly_seasonal.std()
        })

        # Quarterly patterns
        quarterly_avg = price_series.groupby(price_series.index.quarter).mean()
        quarterly_seasonal = price_series.index.quarter.map(quarterly_avg)
        features.append({
            'feature_name': f'{commodity_type}_quarterly_seasonal',
            'values': (quarterly_seasonal - quarterly_seasonal.mean()) / quarterly_seasonal.std()
        })

        # Day of week effects
        dow_avg = price_series.groupby(price_series.index.dayofweek).mean()
        dow_seasonal = price_series.index.dayofweek.map(dow_avg)
        features.append({
            'feature_name': f'{commodity_type}_dow_seasonal',
            'values': (dow_seasonal - dow_seasonal.mean()) / dow_seasonal.std()
        })

        # Holiday proximity (simplified)
        # In production: Use proper holiday calendar
        holiday_proximity = self._calculate_holiday_proximity(price_series.index)
        features.append({
            'feature_name': f'{commodity_type}_holiday_proximity',
            'values': holiday_proximity
        })

        return self._combine_features(features)

    def build_storage_features(
        self,
        prices: pd.Series,
        inventory_data: Optional[pd.Series] = None,
        storage_costs: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Build features related to storage economics and inventory.

        Features:
        - Inventory levels (normalized)
        - Storage cost curves
        - Contango/backwardation indicators
        - Convenience yield estimates
        """
        features = []

        # Inventory levels (if available)
        if inventory_data is not None:
            # Normalize inventory levels
            inv_normalized = (inventory_data - inventory_data.mean()) / inventory_data.std()
            features.append({
                'feature_name': 'inventory_level_normalized',
                'values': inv_normalized.fillna(0)
            })

            # Inventory trend (momentum)
            inv_trend = inventory_data.diff(30)  # 30-day change
            features.append({
                'feature_name': 'inventory_trend_30d',
                'values': inv_trend.fillna(0)
            })

        # Forward curve structure (contango/backwardation)
        if len(prices) > 30:
            # Calculate term structure
            front_month = prices
            back_months = prices.shift(-30)  # Approximate 1-month forward

            term_structure = (back_months / front_month - 1) * 100  # Percentage
            features.append({
                'feature_name': 'term_structure_pct',
                'values': term_structure.fillna(0)
            })

        # Storage cost estimates (if available)
        if storage_costs is not None:
            # Storage economics
            storage_yield = (storage_costs / prices) * 100  # As percentage of price
            features.append({
                'feature_name': 'storage_cost_pct',
                'values': storage_yield.fillna(0)
            })

        return self._combine_features(features)

    def build_generation_spread_features(
        self,
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

        features: Dict[str, float] = {}

        if power_prices is None or power_prices.empty:
            return features

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

        # Spark spread metrics
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

        # Dark spread metrics
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

        # Capacity utilization metrics
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
                    capacity_df["capacity_headroom"] = capacity_df["capacity"] - capacity_df["load"]

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

    def build_weather_demand_features(
        self,
        prices: pd.Series,
        temperature_data: Optional[pd.Series] = None,
        demand_data: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Build features related to weather and demand patterns.

        Features:
        - Temperature deviations
        - Heating/cooling degree days
        - Demand elasticity measures
        - Weather-related volatility
        """
        features = []

        if temperature_data is not None:
            # Temperature deviations from normal
            temp_normal = temperature_data.rolling(365, center=True).mean()
            temp_deviation = temperature_data - temp_normal
            features.append({
                'feature_name': 'temperature_deviation',
                'values': temp_deviation.fillna(0)
            })

            # Heating Degree Days (HDD)
            hdd = np.maximum(65 - temperature_data, 0)  # Base temperature 65°F
            features.append({
                'feature_name': 'heating_degree_days',
                'values': hdd.fillna(0)
            })

            # Cooling Degree Days (CDD)
            cdd = np.maximum(temperature_data - 65, 0)  # Base temperature 65°F
            features.append({
                'feature_name': 'cooling_degree_days',
                'values': cdd.fillna(0)
            })

        if demand_data is not None:
            # Demand growth rates
            demand_growth = demand_data.pct_change(30)  # Month-over-month growth
            features.append({
                'feature_name': 'demand_growth_30d',
                'values': demand_growth.fillna(0)
            })

            # Price-demand elasticity (correlation)
            if len(prices) > 30:
                price_demand_corr = prices.rolling(90).corr(demand_data)
                features.append({
                    'feature_name': 'price_demand_correlation',
                    'values': price_demand_corr.fillna(0)
                })

        return self._combine_features(features)

    def build_volatility_features(
        self,
        prices: pd.Series,
        horizons: List[int] = [5, 10, 30, 90]
    ) -> pd.DataFrame:
        """
        Build volatility-based features.

        Features:
        - Realized volatility over different horizons
        - Volatility regime indicators
        - Jump detection
        - Volatility clustering measures
        """
        features = []

        for horizon in horizons:
            # Realized volatility
            returns = prices.pct_change()
            vol = returns.rolling(horizon).std() * np.sqrt(252)  # Annualized
            features.append({
                'feature_name': f'realized_vol_{horizon}d',
                'values': vol.fillna(0)
            })

            # Volatility regime (high/low)
            vol_regime = (vol > vol.rolling(90).quantile(0.75)).astype(int)
            features.append({
                'feature_name': f'vol_regime_{horizon}d',
                'values': vol_regime
            })

        # Volatility of volatility (vol-of-vol)
        if len(prices) > 90:
            returns = prices.pct_change()
            vol_series = returns.rolling(30).std()
            vol_of_vol = vol_series.rolling(90).std()
            features.append({
                'feature_name': 'volatility_of_volatility',
                'values': vol_of_vol.fillna(0)
            })

        return self._combine_features(features)

    def _calculate_holiday_proximity(self, dates: pd.Index) -> pd.Series:
        """Calculate proximity to major holidays."""
        # Simplified holiday proximity calculation
        # In production: Use proper holiday calendar

        holiday_dates = [
            datetime(dates[0].year, 1, 1),   # New Year
            datetime(dates[0].year, 7, 4),   # Independence Day
            datetime(dates[0].year, 12, 25), # Christmas
        ]

        proximity = []
        for date in dates:
            min_distance = min(abs((date - holiday).days) for holiday in holiday_dates)
            proximity.append(max(0, 30 - min_distance) / 30)  # Normalize to 0-1

        return pd.Series(proximity, index=dates)

    def _combine_features(self, feature_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Combine multiple feature series into a single DataFrame."""
        if not feature_list:
            return pd.DataFrame()

        # Start with the first feature's index
        result_df = pd.DataFrame(index=feature_list[0]['values'].index)

        for feature in feature_list:
            result_df[feature['feature_name']] = feature['values']

        return result_df.fillna(0)
