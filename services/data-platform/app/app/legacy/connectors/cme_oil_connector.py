"""
CME Group Oil Futures Connector

Overview
--------
Publishes crude oil futures curves (e.g., WTI, Brent) with contract metadata
and rollover logic. This scaffold emits representative values for development;
integrate with CME feeds for production.

Data Flow
---------
CME feed → normalize contracts → canonical curve events → Kafka

Configuration
-------------
- Contracts registered in helper (`_register_*`) with exchange/unit details.
- `kafka.topic`/`kafka.bootstrap_servers` for event emission.

Operational Notes
-----------------
- Use base `BaseCommodityConnector` helpers for curve emission and rollover
  evaluation to ensure consistent mapping across commodity connectors.
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json
from decimal import Decimal

from .commodities.base import BaseCommodityConnector
from .base import CommodityType, ContractSpecification, FuturesContract

logger = logging.getLogger(__name__)


class CMEGroupConnector(BaseCommodityConnector):
    """
    CME Group oil futures data connector.

    Responsibilities:
    - Ingest WTI and Brent crude futures
    - Handle contract rollover logic
    - Map CME data to canonical schema
    - Support real-time and historical data
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # CME API configuration
        self.api_base_url = config.get("api_base_url", "https://www.cmegroup.com/CmeWS/mdp/v2")
        self.api_key = config.get("api_key")
        self.realtime_enabled = config.get("realtime_enabled", False)

        # Contract specifications for oil futures
        self._register_contract_specifications()

    def _register_contract_specifications(self) -> None:
        """Register contract specifications for oil futures."""

        # WTI Crude Oil Futures (CL)
        wti_spec = ContractSpecification(
            commodity_code="CL",
            commodity_type=CommodityType.OIL,
            contract_unit="barrels",
            quality_spec={
                "grade": "Light Sweet Crude Oil",
                "api_gravity": "37-42 degrees",
                "sulfur_content": "0.42% maximum",
                "delivery_points": ["Cushing, OK"]
            },
            delivery_location="Cushing, OK",
            exchange="CME",
            contract_size=1000.0,  # 1000 barrels
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(wti_spec)

        # Brent Crude Oil Futures (BZ)
        brent_spec = ContractSpecification(
            commodity_code="BZ",
            commodity_type=CommodityType.OIL,
            contract_unit="barrels",
            quality_spec={
                "grade": "Brent Crude Oil",
                "api_gravity": "38.3 degrees",
                "sulfur_content": "0.37%",
                "delivery_points": ["North Sea"]
            },
            delivery_location="North Sea",
            exchange="ICE",
            contract_size=1000.0,
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(brent_spec)

        # RBOB Gasoline Futures (RB)
        rbob_spec = ContractSpecification(
            commodity_code="RB",
            commodity_type=CommodityType.REFINED_PRODUCTS,
            contract_unit="gallons",
            quality_spec={
                "grade": "RBOB Gasoline",
                "octane_rating": "87 AKI minimum",
                "sulfur_content": "80 ppm maximum",
                "delivery_points": ["New York Harbor"]
            },
            delivery_location="New York Harbor",
            exchange="CME",
            contract_size=42000.0,  # 42,000 gallons
            tick_size=0.0001,
            currency="USD"
        )
        self.register_contract_specification(rbob_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available oil futures data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "contracts": [
                {
                    "commodity_code": "CL",
                    "name": "WTI Crude Oil Futures",
                    "exchange": "CME",
                    "contract_size": 1000,
                    "unit": "barrels",
                    "currency": "USD",
                    "tick_size": 0.01
                },
                {
                    "commodity_code": "BZ",
                    "name": "Brent Crude Oil Futures",
                    "exchange": "ICE",
                    "contract_size": 1000,
                    "unit": "barrels",
                    "currency": "USD",
                    "tick_size": 0.01
                },
                {
                    "commodity_code": "RB",
                    "name": "RBOB Gasoline Futures",
                    "exchange": "CME",
                    "contract_size": 42000,
                    "unit": "gallons",
                    "currency": "USD",
                    "tick_size": 0.0001
                }
            ],
            "data_types": ["settlement_prices", "volume_oi", "real_time_quotes"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull oil futures data from CME Group.

        For production: Use CME MDP 3.0 API for real-time data
        For development: Use public historical data endpoints
        """
        try:
            # Get current trading session
            trading_date = datetime.now(timezone.utc).date()

            # Pull futures curves for each contract
            for commodity_code in ["CL", "BZ", "RB"]:
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
                            exchange=self.get_contract_specification(commodity_code).exchange
                        ) for contract in futures_data
                    ])

                except Exception as e:
                    logger.error(f"Error fetching futures data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in CME oil connector: {e}")
            raise

    def _fetch_futures_curve(self, commodity_code: str, trading_date: date) -> List[Dict[str, Any]]:
        """Fetch futures curve data for a commodity."""
        # For development: Return mock data
        # In production: Use CME MDP API or public data feeds

        spec = self.get_contract_specification(commodity_code)
        if not spec:
            raise ValueError(f"No specification found for {commodity_code}")

        # Mock futures curve data
        mock_contracts = []
        base_price = 80.0 if commodity_code == "CL" else 85.0 if commodity_code == "BZ" else 2.50

        for i in range(1, 13):  # Next 12 months
            contract_month = trading_date.replace(month=trading_date.month + i)
            if contract_month.month > 12:
                contract_month = contract_month.replace(year=contract_month.year + 1, month=contract_month.month - 12)

            # Generate realistic price curve
            months_ahead = i
            price = base_price + (months_ahead * 0.5) + (months_ahead ** 2 * 0.1)

            mock_contracts.append({
                "contract_month": contract_month,
                "settlement_price": round(price, 2),
                "open_interest": int(100000 * (1 / months_ahead)),  # Decreasing OI
                "volume": int(50000 * (1 / months_ahead)),  # Decreasing volume
            })

        return mock_contracts

    def _fetch_real_time_quotes(self) -> Iterator[Dict[str, Any]]:
        """Fetch real-time quotes for active contracts."""
        # In production: Connect to CME MDP 3.0 for real-time data
        # For now: Return spot price events

        for commodity_code in ["CL", "BZ"]:
            spec = self.get_contract_specification(commodity_code)
            if not spec:
                continue

            # Mock real-time price updates
            base_price = 80.0 if commodity_code == "CL" else 85.0
            price = base_price + (0.1 * (2 * (time.time() % 2) - 1))  # Small random walk

            yield self.create_commodity_price_event(
                commodity_code=commodity_code,
                price=price,
                event_time=datetime.now(timezone.utc),
                price_type="spot",
                volume=1000,
                unit=f"USD/{spec.contract_unit}"
            )

    def handle_contract_rollover(self, commodity_code: str, current_contract: str, next_contract: str) -> None:
        """Handle contract rollover for oil futures."""
        logger.info(f"Rolling over {commodity_code} from {current_contract} to {next_contract}")

        # Logic for determining when to roll over
        # Typically: When volume in next contract > current contract
        # Or when next contract is within 2-3 weeks of expiry

        super().handle_contract_rollover(commodity_code, current_contract, next_contract)
