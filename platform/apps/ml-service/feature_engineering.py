"""
Feature engineering for price forecasting models.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from clickhouse_driver import Client
import asyncpg

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Build features for ML models from fundamentals and price history."""
    
    def __init__(self):
        self.ch_client = Client(host="clickhouse", port=9000)
        self.pg_pool = None
    
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
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
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

