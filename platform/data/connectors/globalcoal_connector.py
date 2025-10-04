"""
globalCOAL Coal Price Indices Connector

Ingests coal price indices from globalCOAL (formerly Global Coal):
- API2 (Northwest Europe) - Main European benchmark
- API4 (South Africa) - South African export benchmark
- API5 (Colombia) - Colombian export benchmark
- API6 (Australia) - Australian export benchmark
- NEWC (Newcastle, Australia) - Physical coal prices
- Richards Bay (South Africa) - Physical coal prices
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class GlobalCoalConnector(Ingestor):
    """
    globalCOAL coal price indices connector.

    Responsibilities:
    - Ingest coal price indices from globalCOAL API
    - Handle multiple coal benchmarks and regions
    - Map globalCOAL data to canonical schema
    - Support historical data backfills
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.COAL

        # globalCOAL API configuration
        self.api_base_url = config.get("api_base_url", "https://api.globalcoal.com")
        self.api_key = config.get("api_key")
        self.username = config.get("username")
        self.password = config.get("password")

        # Coal index specifications
        self._register_coal_index_specifications()

    def _register_coal_index_specifications(self) -> None:
        """Register specifications for coal price indices."""

        # API2 (Northwest Europe) - Main European benchmark
        api2_spec = ContractSpecification(
            commodity_code="API2",
            commodity_type=CommodityType.COAL,
            contract_unit="tonne",
            quality_spec={
                "grade": "Steam coal",
                "calorific_value": "6000 kcal/kg",
                "sulfur_content": "1.0% maximum",
                "ash_content": "15% maximum",
                "moisture": "10% maximum",
                "delivery_basis": "CIF ARA"
            },
            delivery_location="ARA (Amsterdam-Rotterdam-Antwerp)",
            exchange="globalCOAL",
            contract_size=1000.0,  # 1000 tonnes
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(api2_spec)

        # API4 (South Africa) - South African export benchmark
        api4_spec = ContractSpecification(
            commodity_code="API4",
            commodity_type=CommodityType.COAL,
            contract_unit="tonne",
            quality_spec={
                "grade": "Steam coal",
                "calorific_value": "5500 kcal/kg",
                "sulfur_content": "1.0% maximum",
                "ash_content": "16% maximum",
                "moisture": "12% maximum",
                "delivery_basis": "FOB Richards Bay"
            },
            delivery_location="Richards Bay, South Africa",
            exchange="globalCOAL",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(api4_spec)

        # NEWC (Newcastle, Australia) - Physical coal prices
        newc_spec = ContractSpecification(
            commodity_code="NEWC",
            commodity_type=CommodityType.COAL,
            contract_unit="tonne",
            quality_spec={
                "grade": "Steam coal",
                "calorific_value": "6300 kcal/kg",
                "sulfur_content": "0.8% maximum",
                "ash_content": "13% maximum",
                "moisture": "8% maximum",
                "delivery_basis": "FOB Newcastle"
            },
            delivery_location="Newcastle, Australia",
            exchange="globalCOAL",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(newc_spec)

        # Richards Bay (South Africa) - Physical coal prices
        rb_spec = ContractSpecification(
            commodity_code="RICHARDS_BAY",
            commodity_type=CommodityType.COAL,
            contract_unit="tonne",
            quality_spec={
                "grade": "Steam coal",
                "calorific_value": "5500 kcal/kg",
                "sulfur_content": "1.0% maximum",
                "ash_content": "16% maximum",
                "moisture": "12% maximum",
                "delivery_basis": "FOB Richards Bay"
            },
            delivery_location="Richards Bay, South Africa",
            exchange="globalCOAL",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(rb_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available coal price index data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "indices": [
                {
                    "commodity_code": "API2",
                    "name": "API2 (Northwest Europe)",
                    "description": "CIF ARA steam coal index",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "benchmark": True
                },
                {
                    "commodity_code": "API4",
                    "name": "API4 (South Africa)",
                    "description": "FOB Richards Bay steam coal index",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "benchmark": True
                },
                {
                    "commodity_code": "NEWC",
                    "name": "NEWC (Newcastle)",
                    "description": "FOB Newcastle steam coal index",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "benchmark": False
                },
                {
                    "commodity_code": "RICHARDS_BAY",
                    "name": "Richards Bay Physical",
                    "description": "FOB Richards Bay physical coal prices",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "benchmark": False
                }
            ],
            "data_types": ["spot_indices", "forward_curves", "physical_prices"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull coal price index data from globalCOAL.

        For production: Use globalCOAL API for real-time data
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull index data for each coal benchmark
            for commodity_code in ["API2", "API4", "NEWC", "RICHARDS_BAY"]:
                try:
                    index_data = self._fetch_coal_index_data(commodity_code, assessment_date)

                    yield self.create_commodity_price_event(
                        commodity_code=commodity_code,
                        price=index_data["price"],
                        event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                        price_type="index",
                        volume=index_data.get("volume"),
                        unit="USD/tonne"
                    )

                except Exception as e:
                    logger.error(f"Error fetching coal data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in globalCOAL connector: {e}")
            raise

    def _fetch_coal_index_data(self, commodity_code: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch coal index data for a specific benchmark."""

        # Mock coal index data for development
        # In production: Query globalCOAL API

        # Coal price relationships (API2 as European benchmark)
        index_premiums = {
            "API2": 0.0,        # European benchmark
            "API4": -15.0,      # South African discount
            "NEWC": 5.0,        # Australian premium for quality
            "RICHARDS_BAY": -10.0  # Physical discount to API4
        }

        base_price = 120.0  # $120/tonne API2
        premium = index_premiums.get(commodity_code, 0.0)

        # Add realistic daily variation
        import random
        price_variation = random.uniform(-3.0, 3.0)  # +/- $3/tonne
        final_price = base_price + premium + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(100000, 500000),  # Typical daily volumes
            "high": round(final_price + random.uniform(1.0, 3.0), 2),
            "low": round(final_price - random.uniform(1.0, 3.0), 2),
            "change": round(price_variation, 2)
        }

    def _authenticate_globalcoal_api(self) -> Dict[str, str]:
        """Authenticate with globalCOAL API."""
        # In production: Implement API key or OAuth authentication
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
