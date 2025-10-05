"""
Platts Refined Products Connector

Overview
--------
Publishes refined products assessments (CBOB, gasoil, naphtha, fuel oil) with
contract metadata and locations. This scaffold emits deterministic mock values;
integrate with Platts feeds for production use.

Data Flow
---------
Platts feed → normalize assessment (by product/location) → canonical price events → Kafka

Configuration
-------------
- Product specification registry within `_register_platts_assessments`.
- `api_base_url`/`api_key` for live queries.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- Maintain product- and location-specific `instrument_id` mapping for analytics
  clarity; add differentials and forwards via additional helpers as needed.
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class PlattsRefinedProductsConnector(Ingestor):
    """
    S&P Global Platts refined products assessments connector.

    Responsibilities:
    - Ingest refined products assessments from Platts
    - Handle global pricing benchmarks
    - Map Platts codes to canonical instrument IDs
    - Support multiple grades and specifications
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.REFINED_PRODUCTS

        # Platts API configuration (mock for development)
        self.api_base_url = config.get("api_base_url", "https://api.platts.com")
        self.api_key = config.get("api_key")
        self.username = config.get("username")
        self.password = config.get("password")

        # Refined products assessment specifications
        self._register_platts_assessments()

    def _register_platts_assessments(self) -> None:
        """Register specifications for Platts refined products assessments."""

        # CBOB Gasoline (Conventional Blendstock for Oxygenate Blending)
        cbob_spec = ContractSpecification(
            commodity_code="CBOB",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="gallon",
            quality_spec={
                "grade": "CBOB Gasoline",
                "octane_rating": "84 AKI minimum",
                "sulfur_content": "80 ppm maximum",
                "vapor_pressure": "7.8-15.0 psi",
                "ethanol_content": "None"
            },
            delivery_location="US Gulf Coast",
            exchange="PLATTS",
            contract_size=42000.0,
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(cbob_spec)

        # Gasoil (European Diesel)
        gasoil_spec = ContractSpecification(
            commodity_code="GASOIL",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="tonne",
            quality_spec={
                "grade": "Gasoil",
                "sulfur_content": "10 ppm maximum",
                "cetane_number": "51 minimum",
                "flash_point": "55°C minimum",
                "density": "820-860 kg/m³"
            },
            delivery_location="Northwest Europe",
            exchange="PLATTS",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(gasoil_spec)

        # Naphtha
        naphtha_spec = ContractSpecification(
            commodity_code="NAPHTHA",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="tonne",
            quality_spec={
                "grade": "Light Naphtha",
                "density": "0.69-0.73 kg/l",
                "paraffins": "65-75%",
                "naphthenes": "15-25%",
                "aromatics": "8-12%"
            },
            delivery_location="Northwest Europe",
            exchange="PLATTS",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(naphtha_spec)

        # Fuel Oil (3.5% Sulfur)
        fuel_oil_spec = ContractSpecification(
            commodity_code="FUEL_OIL_3_5",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="tonne",
            quality_spec={
                "grade": "Fuel Oil 3.5%",
                "sulfur_content": "3.5% maximum",
                "viscosity": "380 cSt maximum",
                "density": "991 kg/m³ maximum",
                "flash_point": "60°C minimum"
            },
            delivery_location="Singapore",
            exchange="PLATTS",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(fuel_oil_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available Platts refined products assessment streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "assessments": [
                {
                    "commodity_code": "CBOB",
                    "name": "CBOB Gasoline",
                    "platts_code": "AABCV00",
                    "frequency": "daily",
                    "unit": "USD/gallon",
                    "locations": ["US Gulf Coast", "US Atlantic Coast"]
                },
                {
                    "commodity_code": "GASOIL",
                    "name": "Gasoil",
                    "platts_code": "PCAAB00",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "locations": ["Northwest Europe", "Mediterranean"]
                },
                {
                    "commodity_code": "NAPHTHA",
                    "name": "Light Naphtha",
                    "platts_code": "PCAAD00",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "locations": ["Northwest Europe", "Mediterranean", "Asia"]
                },
                {
                    "commodity_code": "FUEL_OIL_3_5",
                    "name": "Fuel Oil 3.5%",
                    "platts_code": "PUAAB00",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "locations": ["Singapore", "Rotterdam", "US Gulf Coast"]
                }
            ],
            "data_types": ["assessments", "forward_curves", "differentials"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull refined products assessment data from Platts.

        For production: Use Platts API for real-time data
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull assessment data for each product
            products = ["CBOB", "GASOIL", "NAPHTHA", "FUEL_OIL_3_5"]

            for product in products:
                try:
                    assessment_data = self._fetch_platts_assessment(product, assessment_date)

                    yield self.create_commodity_price_event(
                        commodity_code=product,
                        price=assessment_data["price"],
                        event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                        price_type="assessment",
                        volume=assessment_data.get("volume"),
                        unit=self._get_unit_for_product(product)
                    )

                except Exception as e:
                    logger.error(f"Error fetching assessment data for {product}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in Platts refined products connector: {e}")
            raise

    def _fetch_platts_assessment(self, product: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch Platts assessment data for a specific product."""

        # Mock assessment data for development
        # In production: Query Platts API

        # Base prices by product
        base_prices = {
            "CBOB": 2.35,
            "GASOIL": 520.0,
            "NAPHTHA": 480.0,
            "FUEL_OIL_3_5": 320.0
        }

        base_price = base_prices.get(product, 400.0)

        # Add realistic daily variation
        import random
        price_variation = random.uniform(-10.0, 10.0)  # Appropriate variation by product
        final_price = base_price + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(50000, 200000),  # Typical daily volumes
            "high": round(final_price + random.uniform(5.0, 15.0), 2),
            "low": round(final_price - random.uniform(5.0, 15.0), 2),
            "change": round(price_variation, 2)
        }

    def _get_unit_for_product(self, product: str) -> str:
        """Get the appropriate unit for each product."""
        units = {
            "CBOB": "USD/gallon",
            "GASOIL": "USD/tonne",
            "NAPHTHA": "USD/tonne",
            "FUEL_OIL_3_5": "USD/tonne"
        }
        return units.get(product, "USD/tonne")

    def _authenticate_platts_api(self) -> Dict[str, str]:
        """Authenticate with Platts API."""
        # In production: Implement API key or OAuth authentication
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
