"""
RGGI (Regional Greenhouse Gas Initiative) Connector

Ingests carbon allowance data from the Regional Greenhouse Gas Initiative:
- RGGI CO2 Allowance futures
- RGGI auction results
- State-specific allowance allocations
- Compliance period tracking
- Secondary market trading data
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification, FuturesContract

logger = logging.getLogger(__name__)


class RGGIConnector(Ingestor):
    """
    RGGI carbon market data connector.

    Responsibilities:
    - Ingest RGGI allowance futures from ICE
    - Handle auction results and settlement data
    - Map RGGI data to canonical schema
    - Track state-specific compliance obligations
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.EMISSIONS

        # RGGI API configuration
        self.ice_api_url = config.get("ice_api_url", "https://api.theice.com")
        self.rggi_auction_url = config.get("rggi_auction_url", "https://rggi.org")
        self.api_key = config.get("api_key")

        # RGGI participating states
        self.rggi_states = [
            "Connecticut", "Delaware", "Maine", "Maryland",
            "Massachusetts", "New Hampshire", "New Jersey",
            "New York", "Rhode Island", "Vermont"
        ]

        # RGGI contract specifications
        self._register_rggi_contracts()

    def _register_rggi_contracts(self) -> None:
        """Register RGGI carbon contract specifications."""

        # RGGI CO2 Allowance Futures
        rggi_spec = ContractSpecification(
            commodity_code="RGGI",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "allowance_type": "RGGI CO2 Allowance",
                "compliance_period": "Current (2021-2025)",
                "sector_coverage": "Power sector",
                "states": self.rggi_states,
                "vintage": "Current control period",
                "registry": "RGGI COATS"
            },
            delivery_location="RGGI COATS Registry",
            exchange="ICE",
            contract_size=1000.0,  # 1000 tonnes CO2
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(rggi_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available RGGI carbon market data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "contracts": [
                {
                    "commodity_code": "RGGI",
                    "name": "RGGI CO2 Allowance Futures",
                    "exchange": "ICE",
                    "contract_size": 1000,
                    "unit": "tonnes CO2",
                    "currency": "USD",
                    "tick_size": 0.01,
                    "states": self.rggi_states,
                    "compliance_period": "2021-2025"
                }
            ],
            "data_types": ["futures_prices", "auction_results", "compliance_obligations", "state_allocations"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull RGGI carbon market data from ICE and RGGI sources.

        For production: Use ICE API and RGGI data feeds
        For development: Generate realistic mock data
        """
        try:
            # Get current trading session
            trading_date = datetime.now(timezone.utc).date()

            # Pull futures curves for RGGI contracts
            try:
                futures_data = self._fetch_rggi_futures(trading_date)
                for contract_data in futures_data:
                    yield self.create_futures_curve_event(
                        commodity_code="RGGI",
                        as_of_date=trading_date,
                        contract_data=contract_data
                    )

                # Update futures contracts cache
                self.update_futures_contracts("RGGI", [
                    FuturesContract(
                        contract_month=contract["contract_month"],
                        settlement_price=contract["settlement_price"],
                        open_interest=contract.get("open_interest", 0),
                        volume=contract.get("volume", 0),
                        contract_code=f"RGGI{contract['contract_month'].strftime('%y%m')}",
                        exchange="ICE"
                    ) for contract in futures_data
                ])

            except Exception as e:
                logger.error(f"Error fetching RGGI futures data: {e}")

            # Pull auction results (quarterly auctions)
            try:
                auction_data = self._fetch_rggi_auction_results(trading_date)
                for auction_result in auction_data:
                    yield self._create_auction_event(auction_result, trading_date)

            except Exception as e:
                logger.error(f"Error fetching RGGI auction data: {e}")

        except Exception as e:
            logger.error(f"Error in RGGI connector: {e}")
            raise

    def _fetch_rggi_futures(self, trading_date: date) -> List[Dict[str, Any]]:
        """Fetch RGGI futures curve data."""

        # Mock RGGI futures data for development
        # In production: Query ICE API

        # RGGI prices are typically lower than EU ETS due to regional scope
        base_price = 12.0  # $12/tonne

        # Generate realistic futures curve
        mock_contracts = []
        for i in range(1, 13):  # Next 12 months
            contract_month = trading_date.replace(month=trading_date.month + i)
            if contract_month.month > 12:
                contract_month = contract_month.replace(year=contract_month.year + 1, month=contract_month.month - 12)

            # Generate price curve with slight contango
            months_ahead = i
            contango_factor = 1 + (months_ahead * 0.003)  # 0.3% per month contango

            price = base_price * contango_factor

            mock_contracts.append({
                "contract_month": contract_month,
                "settlement_price": round(price, 2),
                "open_interest": int(50000 * (1 / months_ahead)),  # Decreasing OI
                "volume": int(25000 * (1 / months_ahead)),  # Decreasing volume
            })

        return mock_contracts

    def _fetch_rggi_auction_results(self, trading_date: date) -> List[Dict[str, Any]]:
        """Fetch RGGI auction results."""

        # Mock auction data for development
        # In production: Query RGGI auction results

        # RGGI auctions are held quarterly
        # Generate recent auction results

        auction_results = []

        # Generate last 4 quarterly auctions
        for i in range(1, 5):
            auction_date = trading_date - timedelta(days=i * 90)  # Quarterly

            # RGGI auction volumes and prices
            auction_volume = 15000000  # 15 million allowances
            clearing_price = 12.0 + (i * 0.5)  # Increasing prices over time

            auction_results.append({
                "auction_date": auction_date,
                "clearing_price": round(clearing_price, 2),
                "volume_offered": auction_volume,
                "volume_sold": int(auction_volume * 0.95),  # 95% sell-through
                "bid_to_cover_ratio": 1.8,
                "participants": 50 + i * 2  # Increasing participation
            })

        return auction_results

    def _create_auction_event(self, auction_data: Dict[str, Any], trading_date: date) -> Dict[str, Any]:
        """Create canonical event for RGGI auction results."""

        return {
            "event_time": datetime.combine(auction_data["auction_date"], datetime.min.time(), timezone.utc),
            "arrival_time": datetime.now(timezone.utc),
            "market": self.commodity_type.value,
            "product": "auction",
            "instrument_id": "RGGI_AUCTION",
            "location_code": "RGGI",
            "price_type": "auction_clearing",
            "value": auction_data["clearing_price"],
            "volume": auction_data["volume_sold"],
            "currency": "USD",
            "unit": "USD/tonne",
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "version_id": 1,
            "metadata": json.dumps({
                "auction_date": auction_data["auction_date"].isoformat(),
                "volume_offered": auction_data["volume_offered"],
                "bid_to_cover_ratio": auction_data["bid_to_cover_ratio"],
                "participants": auction_data["participants"]
            })
        }

    def _authenticate_ice_api(self) -> Dict[str, str]:
        """Authenticate with ICE API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _authenticate_rggi_api(self) -> Dict[str, str]:
        """Authenticate with RGGI data sources."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
