"""
CME Biofuels Futures Connector

Ingests biofuels futures data from Chicago Mercantile Exchange (CME):
- Ethanol futures (EH) - Corn-based ethanol
- Biodiesel futures (BD) - Soybean oil-based biodiesel
- Renewable diesel futures (if available)
- DDGS (Dried Distillers Grains with Solubles) futures
- Corn oil futures
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification, FuturesContract

logger = logging.getLogger(__name__)


class CMEBiofuelsConnector(Ingestor):
    """
    CME biofuels futures data connector.

    Responsibilities:
    - Ingest ethanol and biodiesel futures
    - Handle contract rollover logic for biofuels
    - Map CME data to canonical schema
    - Support real-time and historical data
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.BIOFUELS

        # CME API configuration
        self.api_base_url = config.get("api_base_url", "https://www.cmegroup.com/CmeWS/mdp/v2")
        self.api_key = config.get("api_key")
        self.realtime_enabled = config.get("realtime_enabled", False)

        # Biofuels contract specifications
        self._register_biofuels_contract_specifications()

    def _register_biofuels_contract_specifications(self) -> None:
        """Register contract specifications for biofuels futures."""

        # Ethanol Futures (EH)
        ethanol_spec = ContractSpecification(
            commodity_code="ETHANOL",
            commodity_type=CommodityType.BIOFUELS,
            contract_unit="gallon",
            quality_spec={
                "grade": "Denatured Fuel Ethanol",
                "ethanol_content": "92.1% minimum",
                "methanol_content": "0.5% maximum",
                "water_content": "1.0% maximum",
                "acidity": "0.007% maximum"
            },
            delivery_location="Chicago, Illinois",
            exchange="CME",
            contract_size=29000.0,  # 29,000 gallons
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(ethanol_spec)

        # Biodiesel Futures (BD)
        biodiesel_spec = ContractSpecification(
            commodity_code="BIODIESEL",
            commodity_type=CommodityType.BIOFUELS,
            contract_unit="gallon",
            quality_spec={
                "grade": "Biodiesel (B100)",
                "ester_content": "96.5% minimum",
                "methanol_content": "0.2% maximum",
                "water_content": "0.05% maximum",
                "acid_number": "0.5 mg KOH/g maximum"
            },
            delivery_location="Chicago, Illinois",
            exchange="CME",
            contract_size=42000.0,  # 42,000 gallons
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(biodiesel_spec)

        # DDGS Futures (DK)
        ddgs_spec = ContractSpecification(
            commodity_code="DDGS",
            commodity_type=CommodityType.BIOFUELS,
            contract_unit="tonne",
            quality_spec={
                "grade": "Dried Distillers Grains with Solubles",
                "protein_content": "26% minimum",
                "moisture_content": "13% maximum",
                "fat_content": "8% minimum",
                "fiber_content": "9% maximum"
            },
            delivery_location="Chicago, Illinois",
            exchange="CME",
            contract_size=100.0,  # 100 tonnes
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(ddgs_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available biofuels futures data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "contracts": [
                {
                    "commodity_code": "ETHANOL",
                    "name": "Ethanol Futures",
                    "exchange": "CME",
                    "contract_size": 29000,
                    "unit": "gallons",
                    "currency": "USD",
                    "tick_size": 0.0001,
                    "delivery_location": "Chicago, Illinois"
                },
                {
                    "commodity_code": "BIODIESEL",
                    "name": "Biodiesel Futures",
                    "exchange": "CME",
                    "contract_size": 42000,
                    "unit": "gallons",
                    "currency": "USD",
                    "tick_size": 0.0001,
                    "delivery_location": "Chicago, Illinois"
                },
                {
                    "commodity_code": "DDGS",
                    "name": "DDGS Futures",
                    "exchange": "CME",
                    "contract_size": 100,
                    "unit": "tonnes",
                    "currency": "USD",
                    "tick_size": 0.01,
                    "delivery_location": "Chicago, Illinois"
                }
            ],
            "data_types": ["settlement_prices", "volume_oi", "real_time_quotes"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull biofuels futures data from CME.

        For production: Use CME MDP API for real-time data
        For development: Generate realistic mock data
        """
        try:
            # Get current trading session
            trading_date = datetime.now(timezone.utc).date()

            # Pull futures curves for each contract
            for commodity_code in ["ETHANOL", "BIODIESEL", "DDGS"]:
                try:
                    futures_data = self._fetch_futures_curve(commodity_code, trading_date)
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
                            exchange="CME"
                        ) for contract in futures_data
                    ])

                except Exception as e:
                    logger.error(f"Error fetching futures data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in CME biofuels connector: {e}")
            raise

    def _fetch_futures_curve(self, commodity_code: str, trading_date: date) -> List[Dict[str, Any]]:
        """Fetch futures curve data for a biofuels commodity."""

        # Mock futures curve data for development
        # In production: Query CME API

        spec = self.get_contract_specification(commodity_code)
        if not spec:
            raise ValueError(f"No specification found for {commodity_code}")

        # Base prices for biofuels
        base_prices = {
            "ETHANOL": 1.85,     # $1.85/gallon
            "BIODIESEL": 3.20,   # $3.20/gallon
            "DDGS": 185.0        # $185/tonne
        }

        base_price = base_prices.get(commodity_code, 200.0)

        # Generate realistic futures curve
        mock_contracts = []
        for i in range(1, 13):  # Next 12 months
            contract_month = trading_date.replace(month=trading_date.month + i)
            if contract_month.month > 12:
                contract_month = contract_month.replace(year=contract_month.year + 1, month=contract_month.month - 12)

            # Generate price curve with seasonality
            months_ahead = i

            # Biofuels seasonality (winter premium for ethanol due to blending)
            if commodity_code == "ETHANOL":
                if contract_month.month in [12, 1, 2]:  # Winter
                    seasonal_factor = 1.15  # 15% winter premium
                else:
                    seasonal_factor = 1.0
            else:
                seasonal_factor = 1.0

            # Add slight contango
            contango_factor = 1 + (months_ahead * 0.005)  # 0.5% per month

            price = base_price * seasonal_factor * contango_factor

            mock_contracts.append({
                "contract_month": contract_month,
                "settlement_price": round(price, 3),
                "open_interest": int(15000 * (1 / months_ahead)),  # Decreasing OI
                "volume": int(8000 * (1 / months_ahead)),  # Decreasing volume
            })

        return mock_contracts

    def _authenticate_cme_api(self) -> Dict[str, str]:
        """Authenticate with CME API."""
        # In production: Implement OAuth or API key authentication
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
