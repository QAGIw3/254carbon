"""
OPIS Refined Products Pricing Connector

Overview
--------
Publishes refined products pricing (gasoline, diesel, jet, heating oil) across
key U.S. locations and rack types. This scaffold emits deterministic mock
assessments; integrate with OPIS APIs for production.

Data Flow
---------
OPIS API → normalize assessments (by product/location) → canonical price events → Kafka

Configuration
-------------
- Product/location catalogs registered within the connector.
- `api_base_url`/`api_key` for live mode.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- Construct location-specific `instrument_id` to differentiate markets
  unambiguously for downstream analytics.
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class OPISRefinedProductsConnector(Ingestor):
    """
    OPIS refined petroleum products pricing connector.

    Responsibilities:
    - Ingest gasoline, diesel, and jet fuel pricing
    - Handle regional price variations
    - Map OPIS data to canonical schema
    - Support wholesale and retail pricing
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.REFINED_PRODUCTS

        # OPIS API configuration (mock for development)
        self.api_base_url = config.get("api_base_url", "https://api.opisnet.com")
        self.api_key = config.get("api_key")
        self.username = config.get("username")
        self.password = config.get("password")

        # Refined products specifications
        self._register_refined_products_specifications()

    def _register_refined_products_specifications(self) -> None:
        """Register specifications for refined petroleum products."""

        # Regular Gasoline (RBOB)
        rbob_spec = ContractSpecification(
            commodity_code="RBOB",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="gallon",
            quality_spec={
                "grade": "Regular Gasoline",
                "octane_rating": "87 AKI minimum",
                "sulfur_content": "80 ppm maximum",
                "vapor_pressure": "7.8-11.0 psi",
                "ethanol_content": "10% maximum"
            },
            delivery_location="New York Harbor",
            exchange="OPIS",
            contract_size=42000.0,  # 42,000 gallons
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(rbob_spec)

        # Premium Gasoline
        premium_spec = ContractSpecification(
            commodity_code="PREMIUM_GASOLINE",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="gallon",
            quality_spec={
                "grade": "Premium Gasoline",
                "octane_rating": "91 AKI minimum",
                "sulfur_content": "80 ppm maximum",
                "vapor_pressure": "7.8-11.0 psi",
                "ethanol_content": "10% maximum"
            },
            delivery_location="New York Harbor",
            exchange="OPIS",
            contract_size=42000.0,
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(premium_spec)

        # Ultra Low Sulfur Diesel (ULSD)
        ulsd_spec = ContractSpecification(
            commodity_code="ULSD",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="gallon",
            quality_spec={
                "grade": "Ultra Low Sulfur Diesel",
                "sulfur_content": "15 ppm maximum",
                "cetane_number": "40 minimum",
                "flash_point": "125°F minimum",
                "cloud_point": "-10°F maximum"
            },
            delivery_location="New York Harbor",
            exchange="OPIS",
            contract_size=42000.0,
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(ulsd_spec)

        # Jet Fuel/Kerosene
        jet_spec = ContractSpecification(
            commodity_code="JET_FUEL",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="gallon",
            quality_spec={
                "grade": "Jet A/Jet A-1",
                "flash_point": "100°F minimum",
                "freeze_point": "-40°F maximum",
                "sulfur_content": "0.3% maximum",
                "aromatic_content": "25% maximum"
            },
            delivery_location="New York Harbor",
            exchange="OPIS",
            contract_size=42000.0,
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(jet_spec)

        # Heating Oil
        heating_oil_spec = ContractSpecification(
            commodity_code="HEATING_OIL",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="gallon",
            quality_spec={
                "grade": "No. 2 Heating Oil",
                "sulfur_content": "500 ppm maximum",
                "flash_point": "100°F minimum",
                "pour_point": "0°F maximum"
            },
            delivery_location="New York Harbor",
            exchange="OPIS",
            contract_size=42000.0,
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(heating_oil_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available refined products pricing data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "products": [
                {
                    "commodity_code": "RBOB",
                    "name": "Regular Gasoline (RBOB)",
                    "frequency": "daily",
                    "unit": "USD/gallon",
                    "locations": ["New York Harbor", "Gulf Coast", "Los Angeles", "Chicago"]
                },
                {
                    "commodity_code": "PREMIUM_GASOLINE",
                    "name": "Premium Gasoline",
                    "frequency": "daily",
                    "unit": "USD/gallon",
                    "locations": ["New York Harbor", "Gulf Coast", "Los Angeles", "Chicago"]
                },
                {
                    "commodity_code": "ULSD",
                    "name": "Ultra Low Sulfur Diesel",
                    "frequency": "daily",
                    "unit": "USD/gallon",
                    "locations": ["New York Harbor", "Gulf Coast", "Los Angeles", "Chicago"]
                },
                {
                    "commodity_code": "JET_FUEL",
                    "name": "Jet Fuel/Kerosene",
                    "frequency": "daily",
                    "unit": "USD/gallon",
                    "locations": ["New York Harbor", "Gulf Coast", "Los Angeles", "Chicago"]
                },
                {
                    "commodity_code": "HEATING_OIL",
                    "name": "Heating Oil",
                    "frequency": "daily",
                    "unit": "USD/gallon",
                    "locations": ["New York Harbor", "Gulf Coast", "New England"]
                }
            ],
            "data_types": ["spot_prices", "rack_prices", "retail_prices", "terminal_prices"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull refined products pricing data from OPIS.

        For production: Use OPIS API for real-time data
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull pricing data for each product and location
            products = ["RBOB", "PREMIUM_GASOLINE", "ULSD", "JET_FUEL", "HEATING_OIL"]
            locations = ["New York Harbor", "Gulf Coast", "Los Angeles", "Chicago", "New England"]

            for product in products:
                for location in locations:
                    try:
                        pricing_data = self._fetch_product_pricing(product, location, assessment_date)

                        # Create location-specific instrument ID
                        instrument_id = f"{product}_{location.replace(' ', '_').upper()}"

                        yield self.create_commodity_price_event(
                            commodity_code=product,
                            price=pricing_data["price"],
                            event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                            price_type="spot",
                            volume=pricing_data.get("volume"),
                            location_code=location,
                            unit="USD/gallon"
                        )

                    except Exception as e:
                        logger.error(f"Error fetching pricing data for {product} at {location}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error in OPIS refined products connector: {e}")
            raise

    def _fetch_product_pricing(self, product: str, location: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch pricing data for a specific product and location."""

        # Mock pricing data for development
        # In production: Query OPIS API

        # Base prices by product (USD/gallon)
        base_prices = {
            "RBOB": 2.45,
            "PREMIUM_GASOLINE": 2.85,
            "ULSD": 2.60,
            "JET_FUEL": 2.55,
            "HEATING_OIL": 2.35
        }

        base_price = base_prices.get(product, 2.50)

        # Location differentials
        location_premiums = {
            "New York Harbor": 0.15,
            "Gulf Coast": -0.05,
            "Los Angeles": 0.25,
            "Chicago": 0.10,
            "New England": 0.20
        }

        location_premium = location_premiums.get(location, 0.0)

        # Add realistic daily variation
        import random
        price_variation = random.uniform(-0.05, 0.05)  # +/- 5 cents/gallon
        final_price = base_price + location_premium + price_variation

        return {
            "price": round(final_price, 3),
            "volume": random.randint(100000, 500000),  # Typical daily volumes
            "high": round(final_price + random.uniform(0.02, 0.08), 3),
            "low": round(final_price - random.uniform(0.02, 0.08), 3),
            "change": round(price_variation, 3)
        }

    def _authenticate_opis_api(self) -> Dict[str, str]:
        """Authenticate with OPIS API."""
        # In production: Implement API key or OAuth authentication
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
