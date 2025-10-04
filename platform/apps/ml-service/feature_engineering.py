"""Feature engineering for price forecasting models."""
import logging
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncpg
import numpy as np
import pandas as pd
import yaml
from clickhouse_driver import Client

from commodity_feature_engineer import CommodityFeatureEngineer
from data_access import DataAccessLayer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Build features for ML models from fundamentals and price history."""

    _CONFIG_PATH = Path(__file__).resolve().parent / "config" / "commodity_mappings.yaml"
    _MULTIMODAL_CONFIG_PATH = (
        Path(__file__).resolve().parent / "config" / "multimodal_mapping.yaml"
    )

    def __init__(self):
        self.ch_client = Client(host="clickhouse", port=9000)
        self.pg_pool = None
        self._commodity_config = self._load_commodity_config()
        self._multimodal_config = self._load_multimodal_config()
        self._commodity_feature_engineer = CommodityFeatureEngineer()
        self.data_access = DataAccessLayer(ch_client=self.ch_client)

    async def _get_pg_pool(self):
        """Get PostgreSQL connection pool."""
        if self.pg_pool is None:
            self.pg_pool = await asyncpg.create_pool(
                "postgresql://postgres:postgres@postgres:5432/market_intelligence"
            )
        return self.pg_pool
    
    async def build_features(
        self,
        instrument_id: str,
        horizon_months: int,
        custom_features: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Build feature set for forecasting.
        
        Features include:
        - Historical price statistics (lags, moving averages)
        - Fundamental data (load, generation, capacity)
        - Seasonality indicators
        - Calendar features
        - Custom scenario assumptions
        """
        logger.info(f"Building features for {instrument_id}, horizon={horizon_months}")
        
        # Get instrument metadata
        pool = await self._get_pg_pool()
        async with pool.acquire() as conn:
            instrument = await conn.fetchrow(
                "SELECT * FROM pg.instrument WHERE instrument_id = $1",
                instrument_id,
            )
        
        if not instrument:
            raise ValueError(f"Instrument not found: {instrument_id}")

        market = instrument["market"]
        generation_metrics: Dict[str, float] = {}
        if market == "power":
            generation_config = self._resolve_generation_config(instrument)
            generation_metrics = self._build_generation_features(
                instrument_id=instrument_id,
                generation_config=generation_config,
            )

        # Build feature dataframe
        features = []

        for month_ahead in range(1, horizon_months + 1):
            forecast_date = datetime.utcnow() + timedelta(days=30 * month_ahead)
            
            feature_dict = {
                "month_ahead": month_ahead,
                "forecast_date": forecast_date,
                "month": forecast_date.month,
                "quarter": (forecast_date.month - 1) // 3 + 1,
                "year": forecast_date.year,
            }
            
            # Historical price features
            price_features = self._get_price_features(instrument_id)
            feature_dict.update(price_features)
            
            # Fundamental features
            if market == "power":
                fundamental_features = self._get_power_fundamentals(
                    instrument["location_code"],
                    forecast_date,
                )
                feature_dict.update(fundamental_features)
            
            # Seasonality
            feature_dict["sin_month"] = np.sin(2 * np.pi * forecast_date.month / 12)
            feature_dict["cos_month"] = np.cos(2 * np.pi * forecast_date.month / 12)
            
            # Custom features from scenario
            if custom_features:
                feature_dict.update(custom_features)

            if generation_metrics:
                feature_dict.update(generation_metrics)

            features.append(feature_dict)

        return pd.DataFrame(features)

    def _load_commodity_config(self) -> Dict[str, Any]:
        """Load commodity mapping configuration from YAML."""
        if not self._CONFIG_PATH.exists():
            logger.warning(
                "Commodity mapping config not found at %s; using defaults only",
                self._CONFIG_PATH,
            )
            return {}

        try:
            with self._CONFIG_PATH.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except Exception as exc:
            logger.error("Failed to load commodity mapping config: %s", exc)
            return {}

    def _load_multimodal_config(self) -> Dict[str, Any]:
        """Load multimodal mapping configuration from YAML."""
        path = self._MULTIMODAL_CONFIG_PATH
        if not path.exists():
            logger.warning("Multimodal mapping config not found at %s", path)
            return {}

        try:
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load multimodal mapping config: %s", exc)
            return {}

    @staticmethod
    def _deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Recursively merge override dict into base and return a new dict."""
        if not override:
            return base

        merged = deepcopy(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = FeatureEngineer._deep_update(merged[key], value)
            else:
                merged[key] = value
        return merged

    def refresh_multimodal_config(self) -> None:
        """Reload multimodal configuration from disk."""
        self._multimodal_config = self._load_multimodal_config()

    def get_multimodal_config(self) -> Dict[str, Any]:
        """Return a copy of the currently loaded multimodal config."""
        return deepcopy(self._multimodal_config)

    async def build_multimodal_feature_bundle(
        self,
        instrument_ids: Sequence[str],
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        freq: str = "1D",
        max_forward_fill: Optional[int] = 3,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Build aligned multimodal feature frames for the requested instruments."""

        if not instrument_ids:
            return {}

        base_config = self.get_multimodal_config()
        if config_override:
            base_config = self._deep_update(base_config, config_override)

        instrument_map = base_config.get("instruments", {})
        modality_defaults = base_config.get("modalities", {})

        selected = {
            instrument_id: instrument_map.get(instrument_id)
            for instrument_id in instrument_ids
            if instrument_map.get(instrument_id)
        }

        if not selected:
            logger.warning(
                "No multimodal config entries found for instruments: %s",
                instrument_ids,
            )
            return {}

        # Resolve time window
        resolved_end = self._normalize_end_timestamp(end)
        lookbacks: List[int] = []

        for config in selected.values():
            modalities = config.get("modalities", {})
            for modality_name, modality_spec in modalities.items():
                default_spec = modality_defaults.get(modality_name, {})
                lookback_days = modality_spec.get(
                    "lookback_days",
                    default_spec.get("lookback_days", 90),
                )
                lookbacks.append(int(lookback_days))

        max_lookback = max(lookbacks) if lookbacks else 90
        resolved_start = self._normalize_start_timestamp(start, resolved_end, max_lookback)

        bundle: Dict[str, Dict[str, Any]] = {}

        for instrument_id, config in selected.items():
            commodity_key = config.get("commodity", instrument_id)
            commodity_group = config.get("commodity_group", "unknown")
            modalities = config.get("modalities", {})

            modality_frames: Dict[str, pd.DataFrame] = {}

            # Price modality -------------------------------------------------
            price_spec = modalities.get("price", {})
            price_series = price_spec.get("series") or [instrument_id]
            price_lookback = price_spec.get(
                "lookback_days",
                modality_defaults.get("price", {}).get("lookback_days", max_lookback),
            )
            price_start = max(
                resolved_start,
                resolved_end - timedelta(days=int(price_lookback)),
            )
            price_df = self.data_access.get_price_dataframe(
                price_series,
                start=price_start,
                end=resolved_end,
            )
            if not price_df.empty:
                price_df.columns = [f"price::{col}" for col in price_df.columns]
                modality_frames["price"] = price_df
            else:
                modality_frames["price"] = pd.DataFrame()

            # Fundamentals modality ---------------------------------------
            fundamentals_spec = modalities.get("fundamentals", []) or []
            fundamental_frames: List[pd.Series] = []
            for idx, series_spec in enumerate(fundamentals_spec):
                entity = series_spec.get("entity")
                variable = series_spec.get("variable")
                if not entity or not variable:
                    continue
                lookback = series_spec.get(
                    "lookback_days",
                    modality_defaults.get("fundamentals", {}).get("lookback_days", max_lookback),
                )
                series_start = max(
                    resolved_start,
                    resolved_end - timedelta(days=int(lookback)),
                )
                series = self.data_access.get_fundamental_series(
                    entity,
                    variable,
                    start=series_start,
                    end=resolved_end,
                )
                if series.empty:
                    continue
                series = series.sort_index()
                series.name = series_spec.get(
                    "alias",
                    f"fund::{entity.replace('.', '_')}::{variable}",
                )
                fundamental_frames.append(series)

            if fundamental_frames:
                fundamentals_df = pd.concat(fundamental_frames, axis=1)
            else:
                fundamentals_df = pd.DataFrame()
            modality_frames["fundamentals"] = fundamentals_df

            # Weather modality ---------------------------------------------
            weather_spec = modalities.get("weather", []) or []
            weather_frames: List[pd.Series] = []
            for idx, series_spec in enumerate(weather_spec):
                location = series_spec.get("location") or series_spec.get("entity")
                variable = series_spec.get("variable")
                if not location or not variable:
                    continue
                lookback = series_spec.get(
                    "lookback_days",
                    modality_defaults.get("weather", {}).get("lookback_days", max_lookback),
                )
                series_start = max(
                    resolved_start,
                    resolved_end - timedelta(days=int(lookback)),
                )
                series = self.data_access.get_weather_series(
                    location,
                    variable,
                    start=series_start,
                    end=resolved_end,
                )
                if series.empty:
                    continue
                series = series.sort_index()
                series.name = series_spec.get(
                    "alias",
                    f"wx::{location.replace('.', '_')}::{variable}",
                )
                weather_frames.append(series)

            if weather_frames:
                weather_df = pd.concat(weather_frames, axis=1)
            else:
                weather_df = pd.DataFrame()
            modality_frames["weather"] = weather_df

            aligned_frames, modality_masks = self._align_modalities(
                modality_frames,
                start=resolved_start,
                end=resolved_end,
                freq=freq,
                max_forward_fill=max_forward_fill,
            )

            bundle[commodity_key] = {
                "instrument_id": instrument_id,
                "commodity_group": commodity_group,
                "modalities": aligned_frames,
                "masks": modality_masks,
                "time_index": aligned_frames["price"].index,
            }

        return bundle

    def _normalize_end_timestamp(self, end: Optional[datetime]) -> datetime:
        if end is not None:
            return pd.Timestamp(end).to_pydatetime()
        return pd.Timestamp.utcnow().floor("D").to_pydatetime()

    def _normalize_start_timestamp(
        self,
        start: Optional[datetime],
        end: datetime,
        lookback_days: int,
    ) -> datetime:
        if start is not None:
            return pd.Timestamp(start).to_pydatetime()
        return (pd.Timestamp(end) - pd.Timedelta(days=int(lookback_days))).to_pydatetime()

    def _align_modalities(
        self,
        modality_frames: Dict[str, pd.DataFrame],
        *,
        start: datetime,
        end: datetime,
        freq: str,
        max_forward_fill: Optional[int],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Align modality data to a common time index and generate availability masks.
        """

        target_index = pd.date_range(start=start, end=end, freq=freq)
        aligned: Dict[str, pd.DataFrame] = {}
        masks: Dict[str, pd.DataFrame] = {}

        for modality, frame in modality_frames.items():
            if frame is None or frame.empty:
                aligned_frame = pd.DataFrame(index=target_index)
                mask_frame = pd.DataFrame(index=target_index)
            else:
                working = frame.copy()
                working = working[~working.index.duplicated(keep="last")]
                working = working.sort_index()
                working = working.reindex(target_index)
                mask_frame = (~working.isna()).astype(float)
                if max_forward_fill is not None and max_forward_fill > 0:
                    working = working.ffill(limit=max_forward_fill)
                working = working.fillna(method="bfill", limit=1)
                aligned_frame = working.fillna(0.0)

            aligned[modality] = aligned_frame
            masks[modality] = mask_frame

        return aligned, masks

    def _resolve_generation_config(self, instrument: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve commodity feature config for a specific instrument."""
        config = self._commodity_config or {}
        defaults = config.get("defaults", {})

        instrument_id = instrument.get("instrument_id") if isinstance(instrument, dict) else instrument["instrument_id"]
        instrument_attrs = instrument.get("attrs") if isinstance(instrument, dict) else instrument["attrs"]

        resolved = deepcopy(defaults)

        # Region-level override keyed by ISO/market prefix
        region_key = None
        if instrument_id:
            region_key = instrument_id.split(".")[0]
        elif instrument.get("market"):
            region_key = str(instrument["market"]).upper()

        if region_key:
            resolved = self._deep_update(
                resolved,
                config.get("regions", {}).get(region_key, {}),
            )

        # Instrument-level overrides from YAML
        resolved = self._deep_update(
            resolved,
            config.get("instruments", {}).get(instrument_id, {}),
        )

        # pg.instrument.attrs overrides (preferred key: commodity_features)
        attrs_override = {}
        if isinstance(instrument_attrs, dict):
            attrs_override = instrument_attrs.get("commodity_features") or instrument_attrs.get(
                "generation_spreads", {}
            )

        resolved = self._deep_update(resolved, attrs_override)

        return resolved

    def _build_generation_features(
        self,
        instrument_id: str,
        generation_config: Dict[str, Any],
    ) -> Dict[str, float]:
        """Assemble spark/dark spread and capacity features for power instruments."""
        try:
            power_series = self._get_price_series(instrument_id)
            if power_series.empty:
                return {}

            spark_cfg = generation_config.get("spark", {})
            dark_cfg = generation_config.get("dark", {})
            capacity_cfg = generation_config.get("capacity", {})

            gas_series = (
                self._get_price_series(spark_cfg.get("gas_index"))
                if spark_cfg.get("gas_index")
                else None
            )
            coal_series = (
                self._get_price_series(dark_cfg.get("coal_index"))
                if dark_cfg.get("coal_index")
                else None
            )
            carbon_indices = {
                spark_cfg.get("carbon_index"),
                dark_cfg.get("carbon_index"),
            }
            carbon_indices.discard(None)
            carbon_series = None
            if carbon_indices:
                # Prefer the first available carbon index
                carbon_id = next(iter(carbon_indices))
                carbon_series = self._get_price_series(carbon_id)

            capacity_series = self._get_fundamental_series(
                capacity_cfg.get("capacity_series")
            )
            load_series = self._get_fundamental_series(
                capacity_cfg.get("load_series")
            )

            return self._commodity_feature_engineer.build_generation_spread_features(
                power_prices=power_series,
                gas_prices=gas_series,
                coal_prices=coal_series,
                carbon_prices=carbon_series,
                heat_rate_gas=spark_cfg.get("heat_rate"),
                heat_rate_coal=dark_cfg.get("heat_rate"),
                emissions_factor_gas=spark_cfg.get("emissions_factor"),
                emissions_factor_coal=dark_cfg.get("emissions_factor"),
                fallback_capacity=capacity_cfg.get("fallback_capacity_mw"),
                capacity_series=capacity_series,
                load_series=load_series,
            )
        except Exception as exc:
            logger.warning(
                "Failed to build generation features for %s: %s", instrument_id, exc
            )
            return {}

    def _get_price_series(
        self,
        instrument_id: Optional[str],
        days: int = 90,
        price_type: str = "settle",
    ) -> pd.Series:
        """Fetch recent price series for an instrument from ClickHouse."""
        if not instrument_id:
            return pd.Series(dtype=float)

        days = max(int(days), 1)
        query = f"""
            SELECT event_time, value
            FROM ch.market_price_ticks
            WHERE instrument_id = %(instrument_id)s
              AND price_type = %(price_type)s
              AND event_time >= now() - INTERVAL {days} DAY
            ORDER BY event_time
        """

        return self.data_access.get_price_series(
            instrument_id,
            lookback_days=days,
            price_type=price_type,
        )

    def _get_fundamental_series(
        self,
        series_config: Optional[Dict[str, Any]],
        days: int = 90,
    ) -> pd.Series:
        """Fetch fundamentals series from ClickHouse using config mapping."""
        if not series_config or not series_config.get("entity_id") or not series_config.get("variable"):
            return pd.Series(dtype=float)

        days = max(int(days), 1)
        return self.data_access.get_fundamental_series(
            series_config.get("entity_id"),
            series_config.get("variable"),
            lookback_days=days,
        )

    def _get_weather_series(
        self,
        entity_id: Optional[str],
        variable: Optional[str],
        days: int = 180,
    ) -> pd.Series:
        """Fetch weather series from ClickHouse or fundamentals fallback."""
        days = max(int(days), 1)
        return self.data_access.get_weather_series(
            entity_id,
            variable,
            lookback_days=days,
        )
    
    def _get_price_features(self, instrument_id: str) -> Dict[str, float]:
        """Get historical price-based features."""
        query = """
        SELECT 
            avg(value) as price_avg_30d,
            stddevPop(value) as price_std_30d,
            min(value) as price_min_30d,
            max(value) as price_max_30d,
            quantile(0.5)(value) as price_median_30d
        FROM ch.market_price_ticks
        WHERE instrument_id = %(instrument_id)s
          AND event_time >= now() - INTERVAL 30 DAY
          AND price_type = 'settle'
        """
        
        result = self.ch_client.execute(query, {"instrument_id": instrument_id})
        
        if result:
            return {
                "price_avg_30d": float(result[0][0] or 0),
                "price_std_30d": float(result[0][1] or 0),
                "price_min_30d": float(result[0][2] or 0),
                "price_max_30d": float(result[0][3] or 0),
                "price_median_30d": float(result[0][4] or 0),
            }
        
        return {
            "price_avg_30d": 0.0,
            "price_std_30d": 0.0,
            "price_min_30d": 0.0,
            "price_max_30d": 0.0,
            "price_median_30d": 0.0,
        }
    
    def _get_power_fundamentals(
        self,
        location_code: str,
        forecast_date: datetime,
    ) -> Dict[str, float]:
        """Get power market fundamental features."""
        # Extract zone from location_code (e.g., MISO.HUB.INDIANA -> MISO)
        zone = location_code.split(".")[0]
        
        # Query fundamentals
        query = """
        SELECT 
            variable,
            avg(value) as avg_value
        FROM ch.fundamentals_series
        WHERE entity_id = %(zone)s
          AND ts >= now() - INTERVAL 30 DAY
          AND variable IN ('load', 'wind_gen', 'solar_gen', 'gas_price')
        GROUP BY variable
        """
        
        result = self.ch_client.execute(query, {"zone": zone})
        
        fundamentals = {row[0]: float(row[1]) for row in result}
        
        return {
            "load_avg": fundamentals.get("load", 0.0),
            "wind_gen_avg": fundamentals.get("wind_gen", 0.0),
            "solar_gen_avg": fundamentals.get("solar_gen", 0.0),
            "gas_price_avg": fundamentals.get("gas_price", 0.0),
        }
    
    async def build_training_dataset(
        self,
        instrument_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Build training dataset with features and targets.
        
        Returns DataFrame with historical features and realized prices.
        """
        logger.info(
            f"Building training dataset for {instrument_id}: "
            f"{start_date} to {end_date}"
        )
        
        # Get realized prices (targets)
        query = """
        SELECT 
            toDate(event_time) as date,
            avg(value) as realized_price
        FROM ch.market_price_ticks
        WHERE instrument_id = %(instrument_id)s
          AND toDate(event_time) >= %(start_date)s
          AND toDate(event_time) <= %(end_date)s
          AND price_type = 'settle'
        GROUP BY date
        ORDER BY date
        """
        
        result = self.ch_client.execute(
            query,
            {
                "instrument_id": instrument_id,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        
        if not result:
            raise ValueError(f"No training data found for {instrument_id}")
        
        # Build features for each date
        training_data = []
        
        for row in result:
            date_val, price = row
            
            # Build features as of this date
            # In production, would use actual historical fundamentals
            feature_dict = {
                "date": date_val,
                "target_price": float(price),
                "month": date_val.month,
                "quarter": (date_val.month - 1) // 3 + 1,
                "year": date_val.year,
                "sin_month": np.sin(2 * np.pi * date_val.month / 12),
                "cos_month": np.cos(2 * np.pi * date_val.month / 12),
                # Add more features...
            }
            
            training_data.append(feature_dict)
        
        return pd.DataFrame(training_data)
