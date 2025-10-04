"""
Natural Gas Intelligence (NGI) Connector

Ingests natural gas spot market data and regional hub prices from NGI:
- Daily spot prices at major US hubs
- Regional price assessments
- Pipeline flow data and capacity information
- Storage data and injection/withdrawal rates
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class NGIConnector(Ingestor):
    """
    Natural Gas Intelligence spot market data connector.

    Responsibilities:
    - Ingest daily spot prices from major US gas hubs
    - Collect regional price assessments
    - Map NGI data to canonical schema
    - Handle pipeline flow and capacity data
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.GAS

        # NGI API configuration (mock for development)
        self.api_base_url = config.get("api_base_url", "https://api.naturalgasintel.com")
        self.api_key = config.get("api_key")
        self.username = config.get("username")
        self.password = config.get("password")

        # Major US natural gas hubs tracked by NGI
        self._register_gas_hub_specifications()

    def _register_gas_hub_specifications(self) -> None:
        """Register specifications for major US natural gas hubs."""

        # Henry Hub (Louisiana) - Primary benchmark
        hh_spec = ContractSpecification(
            commodity_code="HENRY_HUB",
            commodity_type=CommodityType.GAS,
            contract_unit="MMBtu",
            quality_spec={
                "hub": "Henry Hub",
                "location": "Erath, Louisiana",
                "pipeline_interconnections": ["Texas Eastern", "Transcontinental", "Columbia Gulf"]
            },
            delivery_location="Henry Hub, Louisiana",
            exchange="NGI",
            contract_size=1.0,
            tick_size=0.001,
            currency="USD"
        )
        self.register_contract_specification(hh_spec)

        # Chicago Citygate
        chicago_spec = ContractSpecification(
            commodity_code="CHICAGO_CITYGATE",
            commodity_type=CommodityType.GAS,
            contract_unit="MMBtu",
            quality_spec={
                "hub": "Chicago Citygate",
                "location": "Chicago, Illinois",
                "pipeline_interconnections": ["ANR", "NGPL", "Trunkline"]
            },
            delivery_location="Chicago, Illinois",
            exchange="NGI",
            contract_size=1.0,
            tick_size=0.001,
            currency="USD"
        )
        self.register_contract_specification(chicago_spec)

        # Dominion South (Pennsylvania)
        dominion_spec = ContractSpecification(
            commodity_code="DOMINION_SOUTH",
            commodity_type=CommodityType.GAS,
            contract_unit="MMBtu",
            quality_spec={
                "hub": "Dominion South",
                "location": "Pennsylvania",
                "pipeline_interconnections": ["Dominion Transmission", "Texas Eastern"]
            },
            delivery_location="Pennsylvania",
            exchange="NGI",
            contract_size=1.0,
            tick_size=0.001,
            currency="USD"
        )
        self.register_contract_specification(dominion_spec)

        # SoCal Citygate (California)
        socal_spec = ContractSpecification(
            commodity_code="SOCAL_CITYGATE",
            commodity_type=CommodityType.GAS,
            contract_unit="MMBtu",
            quality_spec={
                "hub": "SoCal Citygate",
                "location": "Southern California",
                "pipeline_interconnections": ["El Paso Natural Gas", "Kern River", "Transwestern"]
            },
            delivery_location="Southern California",
            exchange="NGI",
            contract_size=1.0,
            tick_size=0.001,
            currency="USD"
        )
        self.register_contract_specification(socal_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available natural gas spot market data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "hubs": [
                {
                    "commodity_code": "HENRY_HUB",
                    "name": "Henry Hub",
                    "location": "Erath, Louisiana",
                    "frequency": "daily",
                    "unit": "USD/MMBtu",
                    "benchmark": True
                },
                {
                    "commodity_code": "CHICAGO_CITYGATE",
                    "name": "Chicago Citygate",
                    "location": "Chicago, Illinois",
                    "frequency": "daily",
                    "unit": "USD/MMBtu",
                    "benchmark": False
                },
                {
                    "commodity_code": "DOMINION_SOUTH",
                    "name": "Dominion South",
                    "location": "Pennsylvania",
                    "frequency": "daily",
                    "unit": "USD/MMBtu",
                    "benchmark": False
                },
                {
                    "commodity_code": "SOCAL_CITYGATE",
                    "name": "SoCal Citygate",
                    "location": "Southern California",
                    "frequency": "daily",
                    "unit": "USD/MMBtu",
                    "benchmark": False
                }
            ],
            "data_types": ["spot_prices", "pipeline_flows", "storage_data"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull natural gas spot market data from NGI.

        For production: Use NGI API for real-time data
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull spot prices for each hub
            for commodity_code in ["HENRY_HUB", "CHICAGO_CITYGATE", "DOMINION_SOUTH", "SOCAL_CITYGATE"]:
                try:
                    spot_data = self._fetch_spot_price_data(commodity_code, assessment_date)

                    yield self.create_commodity_price_event(
                        commodity_code=commodity_code,
                        price=spot_data["price"],
                        event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                        price_type="spot",
                        volume=spot_data.get("volume"),
                        unit="USD/MMBtu"
                    )

                except Exception as e:
                    logger.error(f"Error fetching spot data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in NGI connector: {e}")
            raise

    def _fetch_spot_price_data(self, commodity_code: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch spot price data for a specific hub and date."""

        # Mock spot price data for development
        # In production: Query NGI API

        # Base prices for different hubs (reflecting typical basis differentials)
        base_prices = {
            "HENRY_HUB": 3.50,        # Benchmark price
            "CHICAGO_CITYGATE": 3.75,  # Midwest premium
            "DOMINION_SOUTH": 3.25,    # Northeast discount
            "SOCAL_CITYGATE": 4.50     # California premium
        }

        base_price = base_prices.get(commodity_code, 3.50)

        # Add realistic daily variation
        import random
        price_variation = random.uniform(-0.15, 0.15)  # +/- 15 cents
        final_price = base_price + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(100000, 500000),  # Typical daily volumes
            "high": round(final_price + random.uniform(0.05, 0.15), 2),
            "low": round(final_price - random.uniform(0.05, 0.15), 2),
            "change": round(price_variation, 2)
        }

    def _authenticate_ngi_api(self) -> Dict[str, str]:
        """Authenticate with NGI API."""
        # In production: Implement API key or OAuth authentication
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
