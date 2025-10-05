"""
Argus/McCloskey Coal Assessments Connector

Overview
--------
Publishes coal price assessments from Argus Media and McCloskey, including
physical prices and regional indices. This scaffold emits deterministic mock
assessments for development; swap to licensed feeds for production.

Data Flow
---------
Provider (Argus/McCloskey) → normalize assessment → canonical price event → Kafka

Configuration
-------------
- `kafka.topic`/`kafka.bootstrap_servers`: Emission settings.
- Contract specifications registered via class setup; extend as needed.

Operational Notes
-----------------
- Use location-specific instrument identifiers for assessments with distinct
  delivery bases (e.g., ARA vs FOB Newcastle) to preserve analytics semantics.
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class ArgusCoalConnector(Ingestor):
    """Connector scaffold for Argus/McCloskey coal assessments."""
    Argus/McCloskey coal price assessments connector.

    Responsibilities:
    - Ingest coal price assessments from Argus Media
    - Handle quality adjustments and specifications
    - Map Argus data to canonical schema
    - Support multiple coal grades and regions
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.COAL

        # Argus API configuration (mock for development)
        self.api_base_url = config.get("api_base_url", "https://api.argusmedia.com")
        self.api_key = config.get("api_key")
        self.username = config.get("username")
        self.password = config.get("password")

        # Coal assessment specifications
        self._register_coal_assessment_specifications()

    def _register_coal_assessment_specifications(self) -> None:
        """Register specifications for coal price assessments."""

        # Newcastle (Australia) - Premium coal
        newcastle_spec = ContractSpecification(
            commodity_code="NEWCASTLE_PREMIUM",
            commodity_type=CommodityType.COAL,
            contract_unit="tonne",
            quality_spec={
                "grade": "Premium hard coking coal",
                "calorific_value": "7200 kcal/kg",
                "sulfur_content": "0.6% maximum",
                "ash_content": "10% maximum",
                "volatile_matter": "22-28%",
                "delivery_basis": "FOB Newcastle"
            },
            delivery_location="Newcastle, Australia",
            exchange="Argus",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(newcastle_spec)

        # Qinhuangdao (China) - Domestic coal
        qhd_spec = ContractSpecification(
            commodity_code="QINHUANGDAO",
            commodity_type=CommodityType.COAL,
            contract_unit="tonne",
            quality_spec={
                "grade": "Steam coal",
                "calorific_value": "5500 kcal/kg",
                "sulfur_content": "1.0% maximum",
                "ash_content": "20% maximum",
                "delivery_basis": "FOB Qinhuangdao"
            },
            delivery_location="Qinhuangdao, China",
            exchange="Argus",
            contract_size=1000.0,
            tick_size=0.01,
            currency="CNY"
        )
        self.register_contract_specification(qhd_spec)

        # Guangzhou (China) - Import coal
        guangzhou_spec = ContractSpecification(
            commodity_code="GUANGZHOU",
            commodity_type=CommodityType.COAL,
            contract_unit="tonne",
            quality_spec={
                "grade": "Steam coal",
                "calorific_value": "5500 kcal/kg",
                "sulfur_content": "1.0% maximum",
                "ash_content": "15% maximum",
                "delivery_basis": "CIF Guangzhou"
            },
            delivery_location="Guangzhou, China",
            exchange="Argus",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(guangzhou_spec)

        # Indonesian coal
        indonesian_spec = ContractSpecification(
            commodity_code="INDONESIAN_COAL",
            commodity_type=CommodityType.COAL,
            contract_unit="tonne",
            quality_spec={
                "grade": "Steam coal",
                "calorific_value": "4200 kcal/kg",
                "sulfur_content": "0.8% maximum",
                "ash_content": "6% maximum",
                "delivery_basis": "FOB Kalimantan"
            },
            delivery_location="Kalimantan, Indonesia",
            exchange="Argus",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(indonesian_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available coal assessment data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "assessments": [
                {
                    "commodity_code": "NEWCASTLE_PREMIUM",
                    "name": "Newcastle Premium Coal",
                    "location": "Australia",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "grade": "Premium hard coking coal"
                },
                {
                    "commodity_code": "QINHUANGDAO",
                    "name": "Qinhuangdao Coal",
                    "location": "China",
                    "frequency": "daily",
                    "unit": "CNY/tonne",
                    "grade": "Steam coal"
                },
                {
                    "commodity_code": "GUANGZHOU",
                    "name": "Guangzhou Coal",
                    "location": "China",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "grade": "Steam coal"
                },
                {
                    "commodity_code": "INDONESIAN_COAL",
                    "name": "Indonesian Coal",
                    "location": "Indonesia",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "grade": "Steam coal"
                }
            ],
            "data_types": ["physical_assessments", "quality_adjustments", "freight_inclusive"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull coal assessment data from Argus/McCloskey.

        For production: Use Argus API for real-time data
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull assessment data for each coal type
            for commodity_code in ["NEWCASTLE_PREMIUM", "QINHUANGDAO", "GUANGZHOU", "INDONESIAN_COAL"]:
                try:
                    assessment_data = self._fetch_coal_assessment_data(commodity_code, assessment_date)

                    yield self.create_commodity_price_event(
                        commodity_code=commodity_code,
                        price=assessment_data["price"],
                        event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                        price_type="assessment",
                        volume=assessment_data.get("volume"),
                        unit=self._get_unit_for_coal(commodity_code)
                    )

                except Exception as e:
                    logger.error(f"Error fetching coal data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in Argus coal connector: {e}")
            raise

    def _fetch_coal_assessment_data(self, commodity_code: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch coal assessment data for a specific type."""

        # Mock coal assessment data for development
        # In production: Query Argus/McCloskey APIs

        # Coal price relationships by quality and location
        price_bases = {
            "NEWCASTLE_PREMIUM": 220.0,  # Premium coking coal
            "QINHUANGDAO": 85.0,         # Chinese domestic coal (CNY)
            "GUANGZHOU": 95.0,           # Chinese import coal
            "INDONESIAN_COAL": 65.0      # Indonesian steam coal
        }

        base_price = price_bases.get(commodity_code, 100.0)

        # Add realistic daily variation
        import random
        price_variation = random.uniform(-5.0, 5.0)  # +/- $5/tonne
        final_price = base_price + price_variation

        # Convert CNY to USD for Qinhuangdao
        if commodity_code == "QINHUANGDAO":
            final_price = final_price / 7.0  # Rough CNY/USD conversion

        return {
            "price": round(final_price, 2),
            "volume": random.randint(50000, 200000),  # Typical daily volumes
            "high": round(final_price + random.uniform(2.0, 5.0), 2),
            "low": round(final_price - random.uniform(2.0, 5.0), 2),
            "change": round(price_variation, 2)
        }

    def _get_unit_for_coal(self, commodity_code: str) -> str:
        """Get the appropriate unit for each coal type."""
        units = {
            "NEWCASTLE_PREMIUM": "USD/tonne",
            "QINHUANGDAO": "USD/tonne",  # Converted from CNY
            "GUANGZHOU": "USD/tonne",
            "INDONESIAN_COAL": "USD/tonne"
        }
        return units.get(commodity_code, "USD/tonne")

    def _authenticate_argus_api(self) -> Dict[str, str]:
        """Authenticate with Argus API."""
        # In production: Implement API key or OAuth authentication
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
