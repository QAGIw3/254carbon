"""
REC (Renewable Energy Certificate) Connector

Ingests Renewable Energy Certificate pricing and trading data:
- State-specific REC markets (CA, TX, NE, etc.)
- Class I vs Class II RECs
- Solar RECs (SRECs)
- Wind RECs
- Compliance vs voluntary REC markets
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class RECConnector(Ingestor):
    """
    Renewable Energy Certificate markets connector.

    Responsibilities:
    - Ingest REC pricing from state markets and exchanges
    - Handle compliance and voluntary REC markets
    - Map REC data to canonical schema
    - Track state-specific RPS requirements
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.RENEWABLES

        # REC market data sources
        self.state_rec_urls = {
            "california": "https://www.wregis.org",
            "texas": "https://www.ercot.com",
            "new_england": "https://www.nepool.com",
            "pjm": "https://www.pjm.com",
            "new_york": "https://www.nyserda.ny.gov",
            "midwest": "https://www.misoenergy.org"
        }

        self.api_key = config.get("api_key")

        # REC specifications
        self._register_rec_specifications()

    def _register_rec_specifications(self) -> None:
        """Register specifications for REC markets."""

        # California REC (WREGIS)
        ca_rec_spec = ContractSpecification(
            commodity_code="CA_REC",
            commodity_type=CommodityType.RENEWABLES,
            contract_unit="certificate",
            quality_spec={
                "state": "California",
                "registry": "Western Renewable Energy Generation Information System (WREGIS)",
                "technology": "Solar, wind, geothermal, biomass, small hydro",
                "vintage": "Current compliance year",
                "rps_class": "Class I (RPS-eligible)",
                "eligibility": "California RPS compliance"
            },
            delivery_location="WREGIS Registry",
            exchange="WREGIS",
            contract_size=1.0,  # 1 MWh
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(ca_rec_spec)

        # Texas REC (ERCOT)
        tx_rec_spec = ContractSpecification(
            commodity_code="TX_REC",
            commodity_type=CommodityType.RENEWABLES,
            contract_unit="certificate",
            quality_spec={
                "state": "Texas",
                "registry": "ERCOT",
                "technology": "Wind, solar, biomass, hydro",
                "vintage": "Current compliance year",
                "rps_class": "Class I",
                "eligibility": "Texas RPS compliance"
            },
            delivery_location="ERCOT Registry",
            exchange="ERCOT",
            contract_size=1.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(tx_rec_spec)

        # New England REC
        ne_rec_spec = ContractSpecification(
            commodity_code="NE_REC",
            commodity_type=CommodityType.RENEWABLES,
            contract_unit="certificate",
            quality_spec={
                "region": "New England",
                "registry": "NEPOOL GIS",
                "technology": "Solar, wind, biomass, hydro",
                "vintage": "Current compliance year",
                "rps_class": "Class I and Class II",
                "eligibility": "New England RPS compliance"
            },
            delivery_location="NEPOOL GIS Registry",
            exchange="NEPOOL",
            contract_size=1.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(ne_rec_spec)

        # PJM REC (GATS)
        pjm_rec_spec = ContractSpecification(
            commodity_code="PJM_REC",
            commodity_type=CommodityType.RENEWABLES,
            contract_unit="certificate",
            quality_spec={
                "region": "PJM Interconnection",
                "registry": "Generation Attribute Tracking System (GATS)",
                "technology": "Solar, wind, biomass, hydro",
                "vintage": "Current compliance year",
                "rps_class": "Class I",
                "eligibility": "State RPS compliance in PJM footprint"
            },
            delivery_location="GATS Registry",
            exchange="PJM",
            contract_size=1.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(pjm_rec_spec)

        # Solar REC (SREC)
        srec_spec = ContractSpecification(
            commodity_code="SREC",
            commodity_type=CommodityType.RENEWABLES,
            contract_unit="certificate",
            quality_spec={
                "technology": "Solar photovoltaic",
                "registry": "State-specific (NJ, MD, DE, PA, OH, DC)",
                "vintage": "Current compliance year",
                "rps_class": "Solar carve-out",
                "eligibility": "State solar RPS compliance"
            },
            delivery_location="State SREC Registries",
            exchange="State Markets",
            contract_size=1.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(srec_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available REC market data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "markets": [
                {
                    "commodity_code": "CA_REC",
                    "name": "California REC",
                    "registry": "WREGIS",
                    "frequency": "daily",
                    "unit": "USD/MWh",
                    "state": "California",
                    "rps_class": "Class I"
                },
                {
                    "commodity_code": "TX_REC",
                    "name": "Texas REC",
                    "registry": "ERCOT",
                    "frequency": "daily",
                    "unit": "USD/MWh",
                    "state": "Texas",
                    "rps_class": "Class I"
                },
                {
                    "commodity_code": "NE_REC",
                    "name": "New England REC",
                    "registry": "NEPOOL GIS",
                    "frequency": "daily",
                    "unit": "USD/MWh",
                    "region": "New England",
                    "rps_class": "Class I/II"
                },
                {
                    "commodity_code": "PJM_REC",
                    "name": "PJM REC",
                    "registry": "GATS",
                    "frequency": "daily",
                    "unit": "USD/MWh",
                    "region": "PJM",
                    "rps_class": "Class I"
                },
                {
                    "commodity_code": "SREC",
                    "name": "Solar REC",
                    "registry": "State SREC Markets",
                    "frequency": "daily",
                    "unit": "USD/MWh",
                    "technology": "Solar",
                    "rps_class": "Solar carve-out"
                }
            ],
            "data_types": ["spot_prices", "forward_prices", "trading_volume", "rps_compliance"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull REC market data from state registries and exchanges.

        For production: Use state registry APIs and market data
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull pricing data for each REC market
            rec_markets = ["CA_REC", "TX_REC", "NE_REC", "PJM_REC", "SREC"]

            for rec_market in rec_markets:
                try:
                    pricing_data = self._fetch_rec_pricing(rec_market, assessment_date)

                    yield self.create_commodity_price_event(
                        commodity_code=rec_market,
                        price=pricing_data["price"],
                        event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                        price_type="spot",
                        volume=pricing_data.get("volume"),
                        unit="USD/MWh"
                    )

                except Exception as e:
                    logger.error(f"Error fetching REC data for {rec_market}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in REC connector: {e}")
            raise

    def _fetch_rec_pricing(self, rec_market: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch REC pricing data for a specific market."""

        # Mock REC pricing for development
        # In production: Query state registries and exchanges

        # REC price relationships by market
        base_prices = {
            "CA_REC": 25.0,       # California premium due to strict RPS
            "TX_REC": 15.0,       # Texas moderate pricing
            "NE_REC": 20.0,       # New England moderate-high pricing
            "PJM_REC": 12.0,      # PJM lower pricing due to surplus
            "SREC": 45.0          # Solar premium for carve-outs
        }

        base_price = base_prices.get(rec_market, 20.0)

        # Add realistic daily variation
        import random
        price_variation = random.uniform(-2.0, 2.0)  # +/- $2/MWh
        final_price = base_price + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(10000, 50000),  # Typical daily volumes
            "high": round(final_price + random.uniform(1.0, 3.0), 2),
            "low": round(final_price - random.uniform(1.0, 3.0), 2),
            "change": round(price_variation, 2)
        }

    def _authenticate_state_api(self, state: str) -> Dict[str, str]:
        """Authenticate with state REC registry APIs."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
