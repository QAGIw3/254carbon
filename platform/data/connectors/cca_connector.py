"""
California Carbon Allowances (CCA) Connector

Overview
--------
Publishes CCA market data (futures, auctions, offsets) for the California
Cap-and-Trade Program. This scaffold emits development-friendly payloads;
integrate with exchange/provider APIs and CARB datasets for production.

Data Flow
---------
Exchange/provider (ICE/CARB) → normalize curves/auctions → canonical events → Kafka

Configuration
-------------
- Provider endpoints (ICE) and CARB auction sources when live.
- Contract specifications registered in helper methods.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- Include program nuances (reserve tiers, offset eligibility) in `quality_spec`
  for downstream analytics (compliance cost, inventory valuation).
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification, FuturesContract

logger = logging.getLogger(__name__)


class CCAConnector(Ingestor):
    """CCA allowance connector scaffold with auctions and futures coverage."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.EMISSIONS

        # California Cap-and-Trade API configuration
        self.ice_api_url = config.get("ice_api_url", "https://api.theice.com")
        self.carb_api_url = config.get("carb_api_url", "https://ww2.arb.ca.gov")
        self.api_key = config.get("api_key")

        # California Cap-and-Trade contract specifications
        self._register_cca_contracts()

    def _register_cca_contracts(self) -> None:
        """Register California carbon contract specifications."""

        # California Carbon Allowance Futures
        cca_spec = ContractSpecification(
            commodity_code="CCA",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "allowance_type": "California Carbon Allowance",
                "compliance_period": "Current (2021-2025)",
                "sector_coverage": "Multiple sectors including power, industry, transportation",
                "vintage": "Current compliance period",
                "registry": "California Air Resources Board (CARB)",
                "reserve_tiers": ["Price Containment Reserve", "Allowance Price Containment Reserve"]
            },
            delivery_location="CARB Compliance Instrument Tracking System Service (CITSS)",
            exchange="ICE",
            contract_size=1000.0,  # 1000 tonnes CO2
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(cca_spec)

        # California Offset Credits
        offset_spec = ContractSpecification(
            commodity_code="CCA_OFFSET",
            commodity_type=CommodityType.EMISSIONS,
            contract_unit="tonne",
            quality_spec={
                "credit_type": "California Offset Credit",
                "project_types": ["Forestry", "Urban forestry", "Dairy digesters", "Rice cultivation", "Mine methane"],
                "verification": "CARB-approved verifiers",
                "vintage": "Project vintage year",
                "registry": "CARB CITSS"
            },
            delivery_location="CARB CITSS",
            exchange="CARB",
            contract_size=1.0,  # 1 tonne CO2
            tick_size=0.01,
            currency="USD"
        )
        self.register_contract_specification(offset_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available California carbon market data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "contracts": [
                {
                    "commodity_code": "CCA",
                    "name": "California Carbon Allowance Futures",
                    "exchange": "ICE",
                    "contract_size": 1000,
                    "unit": "tonnes CO2",
                    "currency": "USD",
                    "tick_size": 0.01,
                    "compliance_period": "2021-2025",
                    "reserve_tiers": ["Price Containment Reserve", "Allowance Price Containment Reserve"]
                },
                {
                    "commodity_code": "CCA_OFFSET",
                    "name": "California Offset Credits",
                    "exchange": "CARB",
                    "contract_size": 1,
                    "unit": "tonnes CO2",
                    "currency": "USD",
                    "tick_size": 0.01,
                    "project_types": ["Forestry", "Urban forestry", "Dairy digesters", "Rice cultivation", "Mine methane"]
                }
            ],
            "data_types": ["futures_prices", "auction_results", "offset_trading", "reserve_tier_data"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull California carbon market data from ICE and CARB.

        For production: Use ICE API and CARB data feeds
        For development: Generate realistic mock data
        """
        try:
            # Get current trading session
            trading_date = datetime.now(timezone.utc).date()

            # Pull CCA futures
            try:
                futures_data = self._fetch_cca_futures(trading_date)
                for contract_data in futures_data:
                    yield self.create_futures_curve_event(
                        commodity_code="CCA",
                        as_of_date=trading_date,
                        contract_data=contract_data
                    )

                # Update futures contracts cache
                self.update_futures_contracts("CCA", [
                    FuturesContract(
                        contract_month=contract["contract_month"],
                        settlement_price=contract["settlement_price"],
                        open_interest=contract.get("open_interest", 0),
                        volume=contract.get("volume", 0),
                        contract_code=f"CCA{contract['contract_month'].strftime('%y%m')}",
                        exchange="ICE"
                    ) for contract in futures_data
                ])

            except Exception as e:
                logger.error(f"Error fetching CCA futures data: {e}")

            # Pull offset credit pricing
            try:
                offset_data = self._fetch_offset_credit_pricing(trading_date)

                yield self.create_commodity_price_event(
                    commodity_code="CCA_OFFSET",
                    price=offset_data["price"],
                    event_time=datetime.combine(trading_date, datetime.min.time(), timezone.utc),
                    price_type="spot",
                    volume=offset_data.get("volume"),
                    unit="USD/tonne"
                )

            except Exception as e:
                logger.error(f"Error fetching offset credit data: {e}")

            # Pull auction results (quarterly auctions)
            try:
                auction_data = self._fetch_cca_auction_results(trading_date)
                for auction_result in auction_data:
                    yield self._create_auction_event(auction_result, trading_date)

            except Exception as e:
                logger.error(f"Error fetching CCA auction data: {e}")

        except Exception as e:
            logger.error(f"Error in CCA connector: {e}")
            raise

    def _fetch_cca_futures(self, trading_date: date) -> List[Dict[str, Any]]:
        """Fetch CCA futures curve data."""

        # Mock CCA futures data for development
        # In production: Query ICE API

        # California carbon prices are typically higher than RGGI due to stricter program
        base_price = 28.0  # $28/tonne

        # Generate realistic futures curve
        mock_contracts = []
        for i in range(1, 13):  # Next 12 months
            contract_month = trading_date.replace(month=trading_date.month + i)
            if contract_month.month > 12:
                contract_month = contract_month.replace(year=contract_month.year + 1, month=contract_month.month - 12)

            # Generate price curve with slight backwardation
            months_ahead = i
            backwardation_factor = 1 - (months_ahead * 0.004)  # 0.4% per month backwardation

            price = base_price * backwardation_factor

            mock_contracts.append({
                "contract_month": contract_month,
                "settlement_price": round(price, 2),
                "open_interest": int(100000 * (1 / months_ahead)),  # Decreasing OI
                "volume": int(50000 * (1 / months_ahead)),  # Decreasing volume
            })

        return mock_contracts

    def _fetch_offset_credit_pricing(self, trading_date: date) -> Dict[str, Any]:
        """Fetch California offset credit pricing data."""

        # Mock offset credit pricing for development
        # In production: Query CARB offset registry

        # Offset credits typically trade at a discount to allowances
        base_allowance_price = 28.0
        offset_discount = 0.15  # 15% discount for offset credits

        price = base_allowance_price * (1 - offset_discount)

        # Add realistic variation
        import random
        price_variation = random.uniform(-1.0, 1.0)  # +/- $1/tonne
        final_price = price + price_variation

        return {
            "price": round(final_price, 2),
            "volume": random.randint(10000, 50000),  # Typical daily volumes
            "high": round(final_price + random.uniform(0.5, 1.5), 2),
            "low": round(final_price - random.uniform(0.5, 1.5), 2),
            "change": round(price_variation, 2)
        }

    def _fetch_cca_auction_results(self, trading_date: date) -> List[Dict[str, Any]]:
        """Fetch CCA auction results."""

        # Mock auction data for development
        # In production: Query CARB auction results

        # California auctions are held quarterly
        # Generate recent auction results

        auction_results = []

        # Generate last 4 quarterly auctions
        for i in range(1, 5):
            auction_date = trading_date - timedelta(days=i * 90)  # Quarterly

            # California auction volumes and prices
            auction_volume = 50000000  # 50 million allowances
            clearing_price = 28.0 + (i * 0.8)  # Increasing prices over time

            # Reserve tier prices (higher than auction floor)
            reserve_price = clearing_price * 1.2

            auction_results.append({
                "auction_date": auction_date,
                "clearing_price": round(clearing_price, 2),
                "reserve_price": round(reserve_price, 2),
                "volume_offered": auction_volume,
                "volume_sold": int(auction_volume * 0.98),  # 98% sell-through
                "bid_to_cover_ratio": 2.1,
                "participants": 80 + i * 3  # Increasing participation
            })

        return auction_results

    def _create_auction_event(self, auction_data: Dict[str, Any], trading_date: date) -> Dict[str, Any]:
        """Create canonical event for CCA auction results."""

        return {
            "event_time": datetime.combine(auction_data["auction_date"], datetime.min.time(), timezone.utc),
            "arrival_time": datetime.now(timezone.utc),
            "market": self.commodity_type.value,
            "product": "auction",
            "instrument_id": "CCA_AUCTION",
            "location_code": "California",
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
                "reserve_price": auction_data["reserve_price"],
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

    def _authenticate_carb_api(self) -> Dict[str, str]:
        """Authenticate with CARB API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
