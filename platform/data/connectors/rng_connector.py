"""
RNG (Renewable Natural Gas) Connector

Ingests Renewable Natural Gas pricing and market data:
- RNG production costs and pricing
- LCFS (Low Carbon Fuel Standard) credits
- RINs pricing for RNG pathways
- Pipeline injection and transportation
- Environmental attribute trading
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class RNGConnector(Ingestor):
    """
    Renewable Natural Gas markets connector.

    Responsibilities:
    - Ingest RNG pricing from various sources
    - Handle LCFS and RINs pricing for RNG
    - Track environmental attributes and credits
    - Monitor production capacity and utilization
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.BIOFUELS

        # RNG market data sources
        self.production_data_url = config.get("production_data_url", "https://rngcoalition.org")
        self.lcfs_registry_url = config.get("lcfs_registry_url", "https://lcfs.ca.gov")
        self.api_key = config.get("api_key")

        # RNG specifications
        self._register_rng_specifications()

    def _register_rng_specifications(self) -> None:
        """Register specifications for RNG products and credits."""

        # RNG Physical Gas
        rng_gas_spec = ContractSpecification(
            commodity_code="RNG_GAS",
            commodity_type=CommodityType.BIOFUELS,
            contract_unit="MMBtu",
            quality_spec={
                "grade": "Renewable Natural Gas",
                "carbon_intensity": "Negative or very low",
                "heating_value": "950-1050 Btu/cf",
                "pipeline_quality": "Interchangeable with conventional gas",
                "feedstock": "Landfill gas, dairy manure, food waste, wastewater"
            },
            delivery_location="Pipeline injection points",
            exchange="RNG",
            contract_size=10000.0,  # 10,000 MMBtu
            tick_size=0.001,
            currency="USD"
        )
        self.register_contract_specification(rng_gas_spec)

        # LCFS Credits for RNG
        lcfs_spec = ContractSpecification(
            commodity_code="LCFS_CREDIT",
            commodity_type=CommodityType.BIOFUELS,
            contract_unit="credit",
            quality_spec={
                "type": "Low Carbon Fuel Standard Credit",
                "standard": "California LCFS",
                "carbon_intensity_reduction": "Required threshold",
                "vintage": "Current compliance period"
            },
            delivery_location="California LCFS Registry",
            exchange="LCFS",
            contract_size=1.0,  # 1 metric tonne CO2 equivalent
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(lcfs_spec)

        # RNG Environmental Attributes
        rng_env_spec = ContractSpecification(
            commodity_code="RNG_ENVIRONMENTAL",
            commodity_type=CommodityType.BIOFUELS,
            contract_unit="attribute",
            quality_spec={
                "type": "Environmental Attribute",
                "certification": "RNG-specific environmental benefits",
                "verification": "Third-party verified",
                "vintage": "Production year"
            },
            delivery_location="Environmental Attribute Registry",
            exchange="RNG",
            contract_size=1.0,  # 1 MMBtu of environmental attributes
            tick_size=0.001,
            currency="USD"
        )
        self.register_contract_specification(rng_env_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available RNG market data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "products": [
                {
                    "commodity_code": "RNG_GAS",
                    "name": "Renewable Natural Gas",
                    "frequency": "weekly",
                    "unit": "USD/MMBtu",
                    "feedstocks": ["Landfill gas", "Dairy manure", "Food waste", "Wastewater"]
                },
                {
                    "commodity_code": "LCFS_CREDIT",
                    "name": "LCFS Credit",
                    "frequency": "daily",
                    "unit": "USD/credit",
                    "standard": "California LCFS"
                },
                {
                    "commodity_code": "RNG_ENVIRONMENTAL",
                    "name": "RNG Environmental Attribute",
                    "frequency": "monthly",
                    "unit": "USD/attribute",
                    "certification": "Third-party verified"
                }
            ],
            "data_types": ["spot_prices", "credit_prices", "production_data", "carbon_intensity"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull RNG market data from various sources.

        For production: Use production data APIs and credit registries
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull RNG gas pricing
            try:
                rng_gas_data = self._fetch_rng_gas_pricing(assessment_date)

                yield self.create_commodity_price_event(
                    commodity_code="RNG_GAS",
                    price=rng_gas_data["price"],
                    event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                    price_type="spot",
                    volume=rng_gas_data.get("volume"),
                    unit="USD/MMBtu"
                )

            except Exception as e:
                logger.error(f"Error fetching RNG gas data: {e}")

            # Pull LCFS credit pricing
            try:
                lcfs_data = self._fetch_lcfs_credit_pricing(assessment_date)

                yield self.create_commodity_price_event(
                    commodity_code="LCFS_CREDIT",
                    price=lcfs_data["price"],
                    event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                    price_type="spot",
                    volume=lcfs_data.get("volume"),
                    unit="USD/credit"
                )

            except Exception as e:
                logger.error(f"Error fetching LCFS credit data: {e}")

            # Pull RNG environmental attribute pricing
            try:
                env_data = self._fetch_rng_environmental_pricing(assessment_date)

                yield self.create_commodity_price_event(
                    commodity_code="RNG_ENVIRONMENTAL",
                    price=env_data["price"],
                    event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                    price_type="spot",
                    volume=env_data.get("volume"),
                    unit="USD/attribute"
                )

            except Exception as e:
                logger.error(f"Error fetching RNG environmental data: {e}")

        except Exception as e:
            logger.error(f"Error in RNG connector: {e}")
            raise

    def _fetch_rng_gas_pricing(self, assessment_date: date) -> Dict[str, Any]:
        """Fetch RNG gas pricing data."""

        # Mock RNG gas pricing for development
        # In production: Query RNG production and pricing data sources

        # RNG typically trades at a premium to conventional gas plus environmental value
        conventional_gas = 3.50  # $/MMBtu
        rng_premium = 2.00       # $2.00/MMBtu premium for RNG

        # Add realistic variation
        import random
        price_variation = random.uniform(-0.20, 0.20)  # +/- $0.20/MMBtu
        final_price = conventional_gas + rng_premium + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(100000, 300000),  # Typical weekly volumes
            "high": round(final_price + random.uniform(0.10, 0.30), 2),
            "low": round(final_price - random.uniform(0.10, 0.30), 2),
            "change": round(price_variation, 2)
        }

    def _fetch_lcfs_credit_pricing(self, assessment_date: date) -> Dict[str, Any]:
        """Fetch LCFS credit pricing data."""

        # Mock LCFS credit pricing for development
        # In production: Query California LCFS registry

        # LCFS credits trade based on carbon price differential
        carbon_price = 50.0  # $/tonne CO2
        lcfs_deficit_factor = 1.2  # Factor for credit demand

        base_value = carbon_price * lcfs_deficit_factor

        # Add market dynamics
        import random
        price_variation = random.uniform(-5.0, 5.0)  # +/- $5/credit
        final_price = base_value + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(50000, 200000),  # Typical daily volumes
            "high": round(final_price + random.uniform(2.0, 8.0), 2),
            "low": round(final_price - random.uniform(2.0, 8.0), 2),
            "change": round(price_variation, 2)
        }

    def _fetch_rng_environmental_pricing(self, assessment_date: date) -> Dict[str, Any]:
        """Fetch RNG environmental attribute pricing data."""

        # Mock environmental attribute pricing for development
        # In production: Query environmental attribute registries

        # Environmental attributes value based on carbon and other benefits
        carbon_benefit = 45.0  # $/tonne CO2 equivalent
        other_benefits = 5.0   # $/MMBtu for other environmental benefits

        base_value = carbon_benefit + other_benefits

        # Add market dynamics
        import random
        price_variation = random.uniform(-3.0, 3.0)  # +/- $3/attribute
        final_price = base_value + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(5000, 15000),  # Typical monthly volumes
            "high": round(final_price + random.uniform(1.0, 4.0), 2),
            "low": round(final_price - random.uniform(1.0, 4.0), 2),
            "change": round(price_variation, 2)
        }

    def _authenticate_rng_api(self) -> Dict[str, str]:
        """Authenticate with RNG data sources."""
        # In production: Implement authentication for various RNG data sources
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
