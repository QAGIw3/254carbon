"""
ICE Natural Gas Futures Connector

Overview
--------
Publishes natural gas futures curves for Henry Hub, TTF, NBP, JKM, and other
regional hubs. This scaffold emits representative values for development; wire
to ICE feeds for production.

Data Flow
---------
ICE feed → normalize contracts → canonical curve/price events → Kafka

Configuration
-------------
- `api_base_url`/`api_key`/auth fields for live mode; `realtime_enabled`.
- Contract specifications registered in `_register_gas_contract_specifications`.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- Use base helpers for event creation/rollover to match other commodity
  connectors and reduce divergence.
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification, FuturesContract

logger = logging.getLogger(__name__)


class ICEGasConnector(Ingestor):
    """
    ICE natural gas futures data connector.

    Responsibilities:
    - Ingest Henry Hub, TTF, NBP, JKM futures
    - Handle contract rollover logic for natural gas
    - Map ICE data to canonical schema
    - Support real-time and historical data
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.GAS

        # ICE API configuration
        self.api_base_url = config.get("api_base_url", "https://www.theice.com/api")
        self.api_key = config.get("api_key")
        self.username = config.get("username")
        self.password = config.get("password")
        self.realtime_enabled = config.get("realtime_enabled", False)

        # Natural gas contract specifications
        self._register_gas_contract_specifications()

    def _register_gas_contract_specifications(self) -> None:
        """Register contract specifications for natural gas futures."""

        # Henry Hub Natural Gas Futures (NG)
        hh_spec = ContractSpecification(
            commodity_code="NG",
            commodity_type=CommodityType.GAS,
            contract_unit="MMBtu",
            quality_spec={
                "pipeline": "Henry Hub",
                "btu_content": "1000-1100 Btu/cf",
                "delivery_point": "Henry Hub, Louisiana"
            },
            delivery_location="Henry Hub, Louisiana",
            exchange="ICE",
            contract_size=10000.0,  # 10,000 MMBtu
            tick_size=0.001,
            currency="USD"
        )
        self.register_contract_specification(hh_spec)

        # TTF Natural Gas Futures (TTF)
        ttf_spec = ContractSpecification(
            commodity_code="TTF",
            commodity_type=CommodityType.GAS,
            contract_unit="MWh",
            quality_spec={
                "delivery_point": "Title Transfer Facility (TTF), Netherlands",
                "quality_range": "G-gas quality"
            },
            delivery_location="TTF, Netherlands",
            exchange="ICE",
            contract_size=1.0,  # 1 MWh
            tick_size=0.001,
            currency="EUR"
        )
        self.register_contract_specification(ttf_spec)

        # NBP Natural Gas Futures (NBP)
        nbp_spec = ContractSpecification(
            commodity_code="NBP",
            commodity_type=CommodityType.GAS,
            contract_unit="therm",
            quality_spec={
                "delivery_point": "National Balancing Point (NBP), UK",
                "quality_range": "UK gas quality specification"
            },
            delivery_location="NBP, UK",
            exchange="ICE",
            contract_size=1000.0,  # 1000 therms
            tick_size=0.001,
            currency="GBP"
        )
        self.register_contract_specification(nbp_spec)

        # JKM LNG Futures (JKM)
        jkm_spec = ContractSpecification(
            commodity_code="JKM",
            commodity_type=CommodityType.GAS,
            contract_unit="MMBtu",
            quality_spec={
                "delivery_window": "LNG cargoes delivered to Japan/Korea",
                "quality": "Standard LNG specification"
            },
            delivery_location="Japan/Korea",
            exchange="ICE",
            contract_size=10000.0,  # 10,000 MMBtu
            tick_size=0.001,
            currency="USD"
        )
        self.register_contract_specification(jkm_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available natural gas futures data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "contracts": [
                {
                    "commodity_code": "NG",
                    "name": "Henry Hub Natural Gas Futures",
                    "exchange": "ICE",
                    "contract_size": 10000,
                    "unit": "MMBtu",
                    "currency": "USD",
                    "tick_size": 0.001,
                    "delivery_location": "Henry Hub, Louisiana"
                },
                {
                    "commodity_code": "TTF",
                    "name": "TTF Natural Gas Futures",
                    "exchange": "ICE",
                    "contract_size": 1,
                    "unit": "MWh",
                    "currency": "EUR",
                    "tick_size": 0.001,
                    "delivery_location": "TTF, Netherlands"
                },
                {
                    "commodity_code": "NBP",
                    "name": "NBP Natural Gas Futures",
                    "exchange": "ICE",
                    "contract_size": 1000,
                    "unit": "therms",
                    "currency": "GBP",
                    "tick_size": 0.001,
                    "delivery_location": "NBP, UK"
                },
                {
                    "commodity_code": "JKM",
                    "name": "JKM LNG Futures",
                    "exchange": "ICE",
                    "contract_size": 10000,
                    "unit": "MMBtu",
                    "currency": "USD",
                    "tick_size": 0.001,
                    "delivery_location": "Japan/Korea"
                }
            ],
            "data_types": ["settlement_prices", "volume_oi", "real_time_quotes"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull natural gas futures data from ICE.

        For production: Use ICE API for real-time data
        For development: Generate realistic mock data
        """
        try:
            # Get current trading session
            trading_date = datetime.now(timezone.utc).date()

            # Pull futures curves for each contract
            for commodity_code in ["NG", "TTF", "NBP", "JKM"]:
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
                            exchange="ICE"
                        ) for contract in futures_data
                    ])

                except Exception as e:
                    logger.error(f"Error fetching futures data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in ICE gas connector: {e}")
            raise

    def _fetch_futures_curve(self, commodity_code: str, trading_date: date) -> List[Dict[str, Any]]:
        """Fetch futures curve data for a natural gas commodity."""

        # Mock futures curve data for development
        # In production: Query ICE API

        spec = self.get_contract_specification(commodity_code)
        if not spec:
            raise ValueError(f"No specification found for {commodity_code}")

        # Base prices for different commodities
        base_prices = {
            "NG": 3.50,    # Henry Hub ~$3.50/MMBtu
            "TTF": 25.0,   # TTF ~€25/MWh
            "NBP": 28.0,   # NBP ~£28/therm
            "JKM": 12.0    # JKM ~$12/MMBtu
        }

        base_price = base_prices.get(commodity_code, 3.50)

        # Generate realistic futures curve
        mock_contracts = []
        for i in range(1, 13):  # Next 12 months
            contract_month = trading_date.replace(month=trading_date.month + i)
            if contract_month.month > 12:
                contract_month = contract_month.replace(year=contract_month.year + 1, month=contract_month.month - 12)

            # Generate price curve with seasonality and contango
            months_ahead = i

            # Add seasonality (winter premium for gas)
            seasonal_factor = 1.0
            if commodity_code in ["NG", "TTF", "NBP"]:  # Northern hemisphere gas
                if contract_month.month in [12, 1, 2]:  # Winter months
                    seasonal_factor = 1.3  # 30% winter premium
                elif contract_month.month in [6, 7, 8]:  # Summer months
                    seasonal_factor = 0.8  # 20% summer discount

            # Add contango (backwardation for some markets)
            contango_factor = 1 + (months_ahead * 0.02)  # 2% per month contango

            price = base_price * seasonal_factor * contango_factor

            # Convert to appropriate units
            if commodity_code == "TTF":
                price = price * 0.293  # Convert €/MWh to €/therm (rough conversion)

            mock_contracts.append({
                "contract_month": contract_month,
                "settlement_price": round(price, 3),
                "open_interest": int(50000 * (1 / months_ahead)),  # Decreasing OI
                "volume": int(25000 * (1 / months_ahead)),  # Decreasing volume
            })

        return mock_contracts

    def _authenticate_ice_api(self) -> Dict[str, str]:
        """Authenticate with ICE API."""
        # In production: Implement OAuth or API key authentication
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
