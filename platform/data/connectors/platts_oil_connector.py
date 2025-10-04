"""
Platts Oil Assessments Connector

Ingests physical oil market assessments from S&P Global Platts for:
- Dated Brent
- Dubai/Oman
- WTI (Cushing, Houston)
- Regional differentials
- Physical crude assessments
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .commodities.base import BaseCommodityConnector
from .base import CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class PlattsOilConnector(BaseCommodityConnector):
    """
    S&P Global Platts oil assessments connector.

    Responsibilities:
    - Ingest physical crude oil assessments
    - Map Platts codes to canonical instrument IDs
    - Handle regional differentials and quality adjustments
    - Support historical backfills
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Platts API configuration (mock for development)
        self.api_base_url = config.get("api_base_url", "https://api.platts.com")
        self.api_key = config.get("api_key")
        self.username = config.get("username")
        self.password = config.get("password")

        # Physical oil assessments to track
        self._register_physical_assessments()

    def _register_physical_assessments(self) -> None:
        """Register physical oil assessment specifications."""

        assessments = [
            {
                "platts_code": "PCAAS00",  # Dated Brent
                "commodity_code": "BRENT_DATED",
                "name": "Dated Brent Crude",
                "quality_spec": {
                    "grade": "Brent Blend",
                    "api_gravity": "38.3 degrees",
                    "sulfur_content": "0.37%"
                },
                "delivery_location": "North Sea",
                "unit": "USD/bbl"
            },
            {
                "platts_code": "PCAAA00",  # Dubai
                "commodity_code": "DUBAI",
                "name": "Dubai Crude",
                "quality_spec": {
                    "grade": "Dubai Crude",
                    "api_gravity": "31 degrees",
                    "sulfur_content": "2.0%"
                },
                "delivery_location": "Persian Gulf",
                "unit": "USD/bbl"
            },
            {
                "platts_code": "AABFZ00",  # WTI Cushing
                "commodity_code": "WTI_CUSHING",
                "name": "WTI Cushing",
                "quality_spec": {
                    "grade": "WTI",
                    "api_gravity": "39.6 degrees",
                    "sulfur_content": "0.24%"
                },
                "delivery_location": "Cushing, OK",
                "unit": "USD/bbl"
            },
            {
                "platts_code": "AABFW00",  # WTI Houston
                "commodity_code": "WTI_HOUSTON",
                "name": "WTI Houston",
                "quality_spec": {
                    "grade": "WTI",
                    "api_gravity": "39.6 degrees",
                    "sulfur_content": "0.24%"
                },
                "delivery_location": "Houston, TX",
                "unit": "USD/bbl"
            }
        ]

        for assessment in assessments:
            spec = ContractSpecification(
                commodity_code=assessment["commodity_code"],
                commodity_type=CommodityType.OIL,
                contract_unit="barrels",
                quality_spec=assessment["quality_spec"],
                delivery_location=assessment["delivery_location"],
                exchange="PLATTS",
                contract_size=1000.0,
                tick_size=0.01,
                currency="USD"
            )
            self.register_contract_specification(spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available physical oil assessments."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "assessments": [
                {
                    "platts_code": "PCAAS00",
                    "commodity_code": "BRENT_DATED",
                    "name": "Dated Brent Crude",
                    "frequency": "daily",
                    "unit": "USD/bbl"
                },
                {
                    "platts_code": "PCAAA00",
                    "commodity_code": "DUBAI",
                    "name": "Dubai Crude",
                    "frequency": "daily",
                    "unit": "USD/bbl"
                },
                {
                    "platts_code": "AABFZ00",
                    "commodity_code": "WTI_CUSHING",
                    "name": "WTI Cushing",
                    "frequency": "daily",
                    "unit": "USD/bbl"
                },
                {
                    "platts_code": "AABFW00",
                    "commodity_code": "WTI_HOUSTON",
                    "name": "WTI Houston",
                    "frequency": "daily",
                    "unit": "USD/bbl"
                }
            ]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull physical oil assessment data from Platts.

        For production: Use Platts API
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull assessments for each tracked commodity
            for commodity_code in ["BRENT_DATED", "DUBAI", "WTI_CUSHING", "WTI_HOUSTON"]:
                try:
                    assessment_data = self._fetch_assessment_data(commodity_code, assessment_date)

                    yield self.create_commodity_price_event(
                        commodity_code=commodity_code,
                        price=assessment_data["price"],
                        event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                        price_type="assessment",
                        volume=assessment_data.get("volume"),
                        unit="USD/bbl"
                    )

                except Exception as e:
                    logger.error(f"Error fetching assessment data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in Platts oil connector: {e}")
            raise

    def _fetch_assessment_data(self, commodity_code: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch assessment data for a specific commodity and date."""

        # Mock assessment data for development
        # In production: Query Platts API

        base_prices = {
            "BRENT_DATED": 82.50,
            "DUBAI": 78.25,
            "WTI_CUSHING": 80.75,
            "WTI_HOUSTON": 81.25
        }

        base_price = base_prices.get(commodity_code, 80.0)

        # Add some realistic daily variation
        import random
        price_variation = random.uniform(-1.5, 1.5)
        final_price = base_price + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(50000, 150000),  # Typical daily volumes
            "high": round(final_price + random.uniform(0.5, 1.5), 2),
            "low": round(final_price - random.uniform(0.5, 1.5), 2)
        }

    def _authenticate_platts_api(self) -> Dict[str, str]:
        """Authenticate with Platts API."""
        # In production: Implement OAuth or API key authentication
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
