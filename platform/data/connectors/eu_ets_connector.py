"""
EU ETS (European Union Emissions Trading System) Connector

Overview
--------
Publishes carbon allowance pricing and futures curves for the EU ETS — covering
EUA, EUAA, and related instruments. This scaffold documents contract metadata
and emits development-friendly series; wire to exchange/provider endpoints for
production.

Data Flow
---------
Exchange/provider (EEX/ICE) → normalize curves/auctions → canonical events → Kafka

Configuration
-------------
- `eex_api_url` / `ice_api_url`: Provider API base URLs.
- `api_key`: Credentials for live queries where required.
- Contract specifications are registered in `_register_eu_ets_contracts`.

Operational Notes
-----------------
- Compliance period and sector coverage live in `quality_spec` and can be used
  downstream to filter instruments for attribution/hedging workflows.
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification, FuturesContract

logger = logging.getLogger(__name__)


class EUETSConnector(Ingestor):
    """
    EU ETS carbon market data connector.

    Responsibilities:
    - Ingest EUA and EUAA futures from EEX and ICE
    - Handle carbon allowance spot and forward prices
    - Map EU ETS data to canonical schema
    - Support compliance period tracking
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.EMISSIONS

        # EU ETS API configuration
        self.eex_api_url = config.get("eex_api_url", "https://api.eex.com")
        self.ice_api_url = config.get("ice_api_url", "https://api.theice.com")
        self.api_key = config.get("api_key")

        # EU ETS contract specifications
        self._register_eu_ets_contracts()

    def _register_eu_ets_contracts(self) -> None:
        """Register EU ETS carbon contract specifications."""

        # EUA Futures (European Union Allowance)
        eua_spec = ContractSpecification(
            commodity_code="EUA",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "allowance_type": "European Union Allowance",
                "compliance_period": "Current (Phase IV: 2021-2030)",
                "sector_coverage": "Power, industry, aviation",
                "vintage": "Current year",
                "registry": "European Union Transaction Log (EUTL)"
            },
            delivery_location="EUTL Registry",
            exchange="EEX/ICE",
            contract_size=1000.0,  # 1000 tonnes CO2
            tick_size=0.01,
            currency="EUR"
        )
        self.register_contract_specification(eua_spec)

        # EUAA Futures (European Union Aviation Allowance)
        euaa_spec = ContractSpecification(
            commodity_code="EUAA",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "allowance_type": "European Union Aviation Allowance",
                "compliance_period": "Current (Phase IV: 2021-2030)",
                "sector_coverage": "Aviation only",
                "vintage": "Current year",
                "registry": "European Union Transaction Log (EUTL)"
            },
            delivery_location="EUTL Registry",
            exchange="EEX/ICE",
            contract_size=1000.0,  # 1000 tonnes CO2
            tick_size=0.01,
            currency="EUR"
        )
        self.register_contract_specification(euaa_spec)

        # CER Futures (Certified Emission Reduction)
        cer_spec = ContractSpecification(
            commodity_code="CER",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "allowance_type": "Certified Emission Reduction",
                "standard": "CDM (Clean Development Mechanism)",
                "project_types": "Various CDM projects",
                "vintage": "Pre-2013 (legacy)",
                "eligibility": "Limited use in EU ETS"
            },
            delivery_location="CDM Registry",
            exchange="ICE",
            contract_size=1000.0,
            tick_size=0.01,
            currency="EUR"
        )
        self.register_contract_specification(cer_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available EU ETS carbon market data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "contracts": [
                {
                    "commodity_code": "EUA",
                    "name": "European Union Allowance Futures",
                    "exchange": "EEX/ICE",
                    "contract_size": 1000,
                    "unit": "tonnes CO2",
                    "currency": "EUR",
                    "tick_size": 0.01,
                    "compliance_period": "Phase IV (2021-2030)"
                },
                {
                    "commodity_code": "EUAA",
                    "name": "European Union Aviation Allowance Futures",
                    "exchange": "EEX/ICE",
                    "contract_size": 1000,
                    "unit": "tonnes CO2",
                    "currency": "EUR",
                    "tick_size": 0.01,
                    "sector": "Aviation"
                },
                {
                    "commodity_code": "CER",
                    "name": "Certified Emission Reduction Futures",
                    "exchange": "ICE",
                    "contract_size": 1000,
                    "unit": "tonnes CO2",
                    "currency": "EUR",
                    "tick_size": 0.01,
                    "standard": "CDM"
                }
            ],
            "data_types": ["futures_prices", "spot_prices", "auction_results", "compliance_data"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull EU ETS carbon market data from EEX and ICE.

        For production: Use exchange APIs for real-time data
        For development: Generate realistic mock data
        """
        try:
            # Get current trading session
            trading_date = datetime.now(timezone.utc).date()

            # Pull futures curves for each contract
            for commodity_code in ["EUA", "EUAA", "CER"]:
                try:
                    futures_data = self._fetch_eu_ets_futures(commodity_code, trading_date)
                    for contract_data in futures_data:
                        yield self.create_futures_curve_event(
                            commodity_code=commodity_code,
                            as_of_date=trading_date,
                            contract_data=contract_data
                        )

                    # Update futures contracts cache
                    self.update_futures_contracts(commodity_code, [
                        FuturesContract(
                            contract_month=contract["contract_month"],
                            settlement_price=contract["settlement_price"],
                            open_interest=contract.get("open_interest", 0),
                            volume=contract.get("volume", 0),
                            contract_code=f"{commodity_code}{contract['contract_month'].strftime('%y%m')}",
                            exchange="EEX/ICE"
                        ) for contract in futures_data
                    ])

                except Exception as e:
                    logger.error(f"Error fetching EU ETS data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in EU ETS connector: {e}")
            raise

    def _fetch_eu_ets_futures(self, commodity_code: str, trading_date: date) -> List[Dict[str, Any]]:
        """Fetch EU ETS futures curve data for a specific contract."""

        # Mock EU ETS futures data for development
        # In production: Query EEX and ICE APIs

        # EUA price curve (EU ETS benchmark)
        if commodity_code == "EUA":
            base_price = 85.0  # €85/tonne
        elif commodity_code == "EUAA":
            base_price = 82.0  # Slightly lower than EUA
        else:  # CER
            base_price = 1.5   # Very low due to limited eligibility

        # Generate realistic futures curve
        mock_contracts = []
        for i in range(1, 13):  # Next 12 months
            contract_month = trading_date.replace(month=trading_date.month + i)
            if contract_month.month > 12:
                contract_month = contract_month.replace(year=contract_month.year + 1, month=contract_month.month - 12)

            # Generate price curve with slight backwardation
            months_ahead = i

            # EU ETS typically shows slight backwardation
            backwardation_factor = 1 - (months_ahead * 0.005)  # 0.5% per month backwardation

            price = base_price * backwardation_factor

            mock_contracts.append({
                "contract_month": contract_month,
                "settlement_price": round(price, 2),
                "open_interest": int(500000 * (1 / months_ahead)),  # Decreasing OI
                "volume": int(250000 * (1 / months_ahead)),  # Decreasing volume
            })

        return mock_contracts

    def _authenticate_eex_api(self) -> Dict[str, str]:
        """Authenticate with EEX API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _authenticate_ice_api(self) -> Dict[str, str]:
        """Authenticate with ICE API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
