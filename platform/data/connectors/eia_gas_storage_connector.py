"""
EIA Natural Gas Storage Connector

Ingests weekly natural gas storage data from the U.S. Energy Information Administration (EIA):
- Weekly storage levels by region (Eastern, Western, Producing)
- Injection and withdrawal rates
- Historical storage data for seasonal analysis
- Storage capacity and utilization rates
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType

logger = logging.getLogger(__name__)


class EIAGasStorageConnector(Ingestor):
    """
    EIA natural gas storage data connector.

    Responsibilities:
    - Ingest weekly storage reports from EIA
    - Parse EIA API data formats
    - Handle historical data backfills
    - Map EIA regions to canonical schema
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.GAS

        # EIA API configuration
        self.eia_api_key = config.get("eia_api_key")
        self.eia_api_base = "https://api.eia.gov/v2"

        # EIA series IDs for natural gas storage
        self._register_eia_series()

    def _register_eia_series(self) -> None:
        """Register EIA series IDs for natural gas storage data."""

        # EIA series IDs for weekly storage data
        self.eia_series = {
            "total_storage": "NG.NW2_EPG0_SWO_R48_BCF.W",  # Total Lower 48 storage
            "eastern_storage": "NG.NW2_EPG0_SWO_R10_BCF.W",  # Eastern storage region
            "western_storage": "NG.NW2_EPG0_SWO_R20_BCF.W",  # Western storage region
            "producing_storage": "NG.NW2_EPG0_SWO_R30_BCF.W",  # Producing storage region
            "salt_storage": "NG.NW2_EPG0_SWO_R41_BCF.W",  # Salt dome storage
            "nonsalt_storage": "NG.NW2_EPG0_SWO_R42_BCF.W",  # Non-salt storage
        }

        # Storage capacity data (updated annually)
        self.capacity_series = {
            "total_capacity": "NG.NW2_EPG0_SAC_R48_BCF.A",  # Total capacity
            "working_capacity": "NG.NW2_EPG0_SWC_R48_BCF.A",  # Working gas capacity
        }

    def discover(self) -> Dict[str, Any]:
        """Discover available EIA storage data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "data_types": [
                {
                    "type": "weekly_storage",
                    "description": "Weekly natural gas storage levels",
                    "frequency": "weekly",
                    "regions": ["Total", "Eastern", "Western", "Producing"],
                    "series_ids": list(self.eia_series.keys())
                },
                {
                    "type": "storage_capacity",
                    "description": "Annual storage capacity data",
                    "frequency": "annual",
                    "series_ids": list(self.capacity_series.keys())
                }
            ]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull EIA storage data.

        For production: Use EIA API for current and historical data
        For development: Generate realistic mock data
        """
        try:
            # Get current date (EIA releases on Thursdays)
            current_date = datetime.now(timezone.utc).date()

            # Find most recent Thursday for EIA data
            days_since_thursday = (current_date.weekday() - 3) % 7  # Thursday = 3
            eia_date = current_date - timedelta(days=days_since_thursday)

            # Pull current storage data
            for series_name, series_id in self.eia_series.items():
                try:
                    storage_data = self._fetch_eia_storage_data(series_id, eia_date)

                    if storage_data:
                        yield self._create_storage_event(series_name, storage_data, eia_date)

                except Exception as e:
                    logger.error(f"Error fetching EIA data for {series_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in EIA storage connector: {e}")
            raise

    def _fetch_eia_storage_data(self, series_id: str, eia_date: date) -> Optional[Dict[str, Any]]:
        """Fetch storage data for a specific EIA series."""

        # Mock EIA storage data for development
        # In production: Query EIA API

        # Generate realistic storage data based on seasonal patterns
        import random

        # Base storage levels (Bcf) - typical ranges
        base_levels = {
            "total_storage": 2500,    # 2,500 Bcf total
            "eastern_storage": 900,   # 900 Bcf eastern
            "western_storage": 400,   # 400 Bcf western
            "producing_storage": 1200,  # 1,200 Bcf producing
            "salt_storage": 800,      # 800 Bcf salt dome
            "nonsalt_storage": 1700   # 1,700 Bcf non-salt
        }

        base_level = base_levels.get(series_id.split('_')[0], 2500)

        # Seasonal variation (higher in fall, lower in spring)
        month = eia_date.month
        if month in [10, 11, 12]:  # Fall injection season
            seasonal_factor = 1.2  # 20% above average
        elif month in [4, 5, 6]:   # Spring withdrawal season
            seasonal_factor = 0.8  # 20% below average
        else:
            seasonal_factor = 1.0

        # Weekly variation
        weekly_variation = random.uniform(-20, 20)  # +/- 20 Bcf

        current_level = base_level * seasonal_factor + weekly_variation

        # Calculate change from previous week
        previous_level = base_level * seasonal_factor + random.uniform(-15, 15)
        net_change = current_level - previous_level

        return {
            "value": round(current_level, 1),
            "net_change": round(net_change, 1),
            "injection": max(0, net_change),
            "withdrawal": abs(min(0, net_change)),
            "as_of_date": eia_date,
            "series_id": series_id
        }

    def _create_storage_event(self, series_name: str, storage_data: Dict[str, Any], eia_date: date) -> Dict[str, Any]:
        """Create canonical storage event from EIA data."""

        # Map series names to regions
        region_mapping = {
            "total_storage": "Total",
            "eastern_storage": "Eastern",
            "western_storage": "Western",
            "producing_storage": "Producing",
            "salt_storage": "Salt",
            "nonsalt_storage": "NonSalt"
        }

        region = region_mapping.get(series_name, "Unknown")

        return {
            "event_time": datetime.combine(eia_date, datetime.min.time(), timezone.utc),
            "arrival_time": datetime.now(timezone.utc),
            "market": self.commodity_type.value,
            "product": "storage",
            "instrument_id": f"storage_{region.lower()}",
            "location_code": region,
            "price_type": "inventory",
            "value": storage_data["value"],
            "volume": storage_data["net_change"],
            "currency": "USD",
            "unit": "Bcf",
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "version_id": 1,
            "metadata": json.dumps({
                "eia_series": storage_data["series_id"],
                "injection": storage_data["injection"],
                "withdrawal": storage_data["withdrawal"],
                "region": region
            })
        }

    def _authenticate_eia_api(self) -> Dict[str, str]:
        """Authenticate with EIA API."""
        return {
            "X-API-Key": self.eia_api_key,
            "Content-Type": "application/json"
        }
