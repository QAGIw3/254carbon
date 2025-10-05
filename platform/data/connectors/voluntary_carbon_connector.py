"""
Voluntary Carbon Markets Connector

Overview
--------
Publishes voluntary carbon credit pricing across major registries and standards
(Gold Standard, Verra VCS, ACR, CAR). This scaffold emits deterministic values
for development; integrate with exchange/registry feeds for production.

Data Flow
---------
Registry/exchange feeds → normalize per-standard → canonical price events → Kafka

Configuration
-------------
- Registry endpoints/keys for live mode.
- Contract specs registered in `_register_voluntary_carbon_contracts`.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- Capture standard, methodology, and project-type metadata in `quality_spec`
  to support downstream comparability and filtering.
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class VoluntaryCarbonConnector(Ingestor):
    """
    Voluntary carbon markets data connector.

    Responsibilities:
    - Ingest voluntary carbon credit pricing
    - Handle multiple registries and standards
    - Map voluntary carbon data to canonical schema
    - Track project types and methodologies
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.EMISSIONS

        # Voluntary carbon market APIs
        self.gold_standard_api = config.get("gold_standard_api", "https://api.goldstandard.org")
        self.verra_api = config.get("verra_api", "https://api.verra.org")
        self.acr_api = config.get("acr_api", "https://api.americancarbonregistry.org")
        self.car_api = config.get("car_api", "https://api.climateactionreserve.org")
        self.api_key = config.get("api_key")

        # Voluntary carbon credit specifications
        self._register_voluntary_carbon_contracts()

    def _register_voluntary_carbon_contracts(self) -> None:
        """Register voluntary carbon credit specifications."""

        # Gold Standard Credits
        gs_spec = ContractSpecification(
            commodity_code="GOLD_STANDARD",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "standard": "Gold Standard",
                "project_types": ["Renewable energy", "Energy efficiency", "Forestry", "Cookstoves"],
                "methodologies": ["GS VER", "GS CER", "GS REDD+"],
                "additionality": "Demonstrated",
                "verification": "Third-party audited",
                "sustainable_development": "Multiple SDG contributions"
            },
            delivery_location="Gold Standard Registry",
            exchange="Gold Standard",
            contract_size=1.0,  # 1 tonne CO2
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(gs_spec)

        # Verra (VCS) Credits
        verra_spec = ContractSpecification(
            commodity_code="VERRA_VCS",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "standard": "Verified Carbon Standard (VCS)",
                "project_types": ["Forestry", "REDD+", "IFM", "ARR", "Energy efficiency"],
                "methodologies": ["VM0001", "VM0007", "VM0010", "VM0012"],
                "additionality": "Demonstrated",
                "verification": "VCS-approved verifiers",
                "registry": "Verra Registry"
            },
            delivery_location="Verra Registry",
            exchange="Verra",
            contract_size=1.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(verra_spec)

        # American Carbon Registry Credits
        acr_spec = ContractSpecification(
            commodity_code="ACR",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "standard": "American Carbon Registry",
                "project_types": ["Forestry", "Agriculture", "Coal mine methane", "Landfill gas"],
                "methodologies": ["ACR IFM", "ACR REDD+", "ACR ACM", "ACR LFG"],
                "additionality": "Demonstrated",
                "verification": "ACR-approved verifiers",
                "registry": "ACR Registry"
            },
            delivery_location="ACR Registry",
            exchange="ACR",
            contract_size=1.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(acr_spec)

        # Climate Action Reserve Credits
        car_spec = ContractSpecification(
            commodity_code="CAR",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "standard": "Climate Action Reserve",
                "project_types": ["Forestry", "Urban forestry", "Livestock", "Mine methane", "Rice cultivation"],
                "methodologies": ["CAR Forest", "CAR Livestock", "CAR Mine Methane"],
                "additionality": "Demonstrated",
                "verification": "CAR-approved verifiers",
                "registry": "CAR Registry"
            },
            delivery_location="CAR Registry",
            exchange="CAR",
            contract_size=1.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(car_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available voluntary carbon market data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "registries": [
                {
                    "commodity_code": "GOLD_STANDARD",
                    "name": "Gold Standard Credits",
                    "registry": "Gold Standard",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "project_types": ["Renewable energy", "Energy efficiency", "Forestry", "Cookstoves"]
                },
                {
                    "commodity_code": "VERRA_VCS",
                    "name": "Verra VCS Credits",
                    "registry": "Verra",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "project_types": ["Forestry", "REDD+", "IFM", "ARR", "Energy efficiency"]
                },
                {
                    "commodity_code": "ACR",
                    "name": "American Carbon Registry Credits",
                    "registry": "ACR",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "project_types": ["Forestry", "Agriculture", "Coal mine methane", "Landfill gas"]
                },
                {
                    "commodity_code": "CAR",
                    "name": "Climate Action Reserve Credits",
                    "registry": "CAR",
                    "frequency": "daily",
                    "unit": "USD/tonne",
                    "project_types": ["Forestry", "Urban forestry", "Livestock", "Mine methane", "Rice cultivation"]
                }
            ],
            "data_types": ["spot_prices", "project_listings", "retirement_data", "methodology_data"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull voluntary carbon market data from various registries.

        For production: Use registry APIs and market data feeds
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull pricing data for each registry
            registries = ["GOLD_STANDARD", "VERRA_VCS", "ACR", "CAR"]

            for registry in registries:
                try:
                    pricing_data = self._fetch_voluntary_carbon_pricing(registry, assessment_date)

                    yield self.create_commodity_price_event(
                        commodity_code=registry,
                        price=pricing_data["price"],
                        event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                        price_type="spot",
                        volume=pricing_data.get("volume"),
                        unit="USD/tonne"
                    )

                except Exception as e:
                    logger.error(f"Error fetching voluntary carbon data for {registry}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in voluntary carbon connector: {e}")
            raise

    def _fetch_voluntary_carbon_pricing(self, registry: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch voluntary carbon pricing data for a specific registry."""

        # Mock voluntary carbon pricing for development
        # In production: Query registry APIs and market platforms

        # Pricing relationships between registries
        base_prices = {
            "GOLD_STANDARD": 12.0,     # Premium for SDG benefits
            "VERRA_VCS": 10.5,         # Large volume, moderate quality
            "ACR": 9.0,               # US-focused, good quality
            "CAR": 8.5                # California-focused, regulatory alignment
        }

        base_price = base_prices.get(registry, 10.0)

        # Add realistic daily variation
        import random
        price_variation = random.uniform(-0.5, 0.5)  # +/- $0.50/tonne
        final_price = base_price + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(50000, 200000),  # Typical daily volumes
            "high": round(final_price + random.uniform(0.2, 0.8), 2),
            "low": round(final_price - random.uniform(0.2, 0.8), 2),
            "change": round(price_variation, 2)
        }

    def _authenticate_gold_standard_api(self) -> Dict[str, str]:
        """Authenticate with Gold Standard API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _authenticate_verra_api(self) -> Dict[str, str]:
        """Authenticate with Verra API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _authenticate_acr_api(self) -> Dict[str, str]:
        """Authenticate with ACR API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _authenticate_car_api(self) -> Dict[str, str]:
        """Authenticate with CAR API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
