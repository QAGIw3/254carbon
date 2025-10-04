"""
European Natural Gas Hubs Connector

Ingests natural gas spot and forward prices from major European hubs:
- TTF (Title Transfer Facility, Netherlands) - Main European benchmark
- NBP (National Balancing Point, UK)
- CEGH (Central European Gas Hub, Austria)
- PSV (Punto di Scambio Virtuale, Italy)
- PEG (Point d'Échange de Gaz, France)
- Zeebrugge Hub (Belgium)
- Gaspool/NCG (Germany)
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class EuropeanGasHubsConnector(Ingestor):
    """
    European natural gas hubs data connector.

    Responsibilities:
    - Ingest spot prices from major European gas hubs
    - Collect forward curve data for TTF and NBP
    - Handle multiple currencies and timezones
    - Map European market structures to canonical schema
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.GAS

        # European gas hub specifications
        self._register_european_hub_specifications()

    def _register_european_hub_specifications(self) -> None:
        """Register specifications for major European natural gas hubs."""

        # TTF (Netherlands) - Main European benchmark
        ttf_spec = ContractSpecification(
            commodity_code="TTF",
            commodity_type=CommodityType.GAS,
            contract_unit="MWh",
            quality_spec={
                "hub": "Title Transfer Facility",
                "location": "Netherlands",
                "quality": "G-gas specification",
                "calorific_value": "35.17 MJ/m³"
            },
            delivery_location="TTF, Netherlands",
            exchange="ICE",
            contract_size=1.0,  # 1 MWh
            tick_size=0.001,
            currency="EUR"
        )
        self.register_contract_specification(ttf_spec)

        # NBP (UK)
        nbp_spec = ContractSpecification(
            commodity_code="NBP",
            commodity_type=CommodityType.GAS,
            contract_unit="therm",
            quality_spec={
                "hub": "National Balancing Point",
                "location": "UK",
                "quality": "UK gas specification"
            },
            delivery_location="NBP, UK",
            exchange="ICE",
            contract_size=1000.0,  # 1000 therms
            tick_size=0.001,
            currency="GBP"
        )
        self.register_contract_specification(nbp_spec)

        # CEGH (Austria)
        cegh_spec = ContractSpecification(
            commodity_code="CEGH",
            commodity_type=CommodityType.GAS,
            contract_unit="MWh",
            quality_spec={
                "hub": "Central European Gas Hub",
                "location": "Austria",
                "quality": "H-gas specification"
            },
            delivery_location="CEGH, Austria",
            exchange="CEGH",
            contract_size=1.0,
            tick_size=0.001,
            currency="EUR"
        )
        self.register_contract_specification(cegh_spec)

        # PSV (Italy)
        psv_spec = ContractSpecification(
            commodity_code="PSV",
            commodity_type=CommodityType.GAS,
            contract_unit="MWh",
            quality_spec={
                "hub": "Punto di Scambio Virtuale",
                "location": "Italy",
                "quality": "Italian gas specification"
            },
            delivery_location="PSV, Italy",
            exchange="PSV",
            contract_size=1.0,
            tick_size=0.001,
            currency="EUR"
        )
        self.register_contract_specification(psv_spec)

        # PEG (France)
        peg_spec = ContractSpecification(
            commodity_code="PEG",
            commodity_type=CommodityType.GAS,
            contract_unit="MWh",
            quality_spec={
                "hub": "Point d'Échange de Gaz",
                "location": "France",
                "quality": "French gas specification"
            },
            delivery_location="PEG, France",
            exchange="PEGAS",
            contract_size=1.0,
            tick_size=0.001,
            currency="EUR"
        )
        self.register_contract_specification(peg_spec)

    def discover(self) -> Dict[str, Any]:
        """Discover available European gas hub data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "hubs": [
                {
                    "commodity_code": "TTF",
                    "name": "Title Transfer Facility",
                    "location": "Netherlands",
                    "frequency": "daily",
                    "unit": "EUR/MWh",
                    "benchmark": True,
                    "timezone": "CET"
                },
                {
                    "commodity_code": "NBP",
                    "name": "National Balancing Point",
                    "location": "UK",
                    "frequency": "daily",
                    "unit": "GBP/therm",
                    "benchmark": True,
                    "timezone": "GMT"
                },
                {
                    "commodity_code": "CEGH",
                    "name": "Central European Gas Hub",
                    "location": "Austria",
                    "frequency": "daily",
                    "unit": "EUR/MWh",
                    "benchmark": False,
                    "timezone": "CET"
                },
                {
                    "commodity_code": "PSV",
                    "name": "Punto di Scambio Virtuale",
                    "location": "Italy",
                    "frequency": "daily",
                    "unit": "EUR/MWh",
                    "benchmark": False,
                    "timezone": "CET"
                },
                {
                    "commodity_code": "PEG",
                    "name": "Point d'Échange de Gaz",
                    "location": "France",
                    "frequency": "daily",
                    "unit": "EUR/MWh",
                    "benchmark": False,
                    "timezone": "CET"
                }
            ],
            "data_types": ["spot_prices", "forward_curves", "intraday_prices"]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull European gas hub data.

        For production: Use exchange APIs and data providers
        For development: Generate realistic mock data
        """
        try:
            # Get current assessment date
            assessment_date = datetime.now(timezone.utc).date()

            # Pull spot prices for each hub
            for commodity_code in ["TTF", "NBP", "CEGH", "PSV", "PEG"]:
                try:
                    spot_data = self._fetch_spot_price_data(commodity_code, assessment_date)

                    yield self.create_commodity_price_event(
                        commodity_code=commodity_code,
                        price=spot_data["price"],
                        event_time=datetime.combine(assessment_date, datetime.min.time(), timezone.utc),
                        price_type="spot",
                        volume=spot_data.get("volume"),
                        unit=self._get_unit_for_hub(commodity_code)
                    )

                except Exception as e:
                    logger.error(f"Error fetching spot data for {commodity_code}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in European gas hubs connector: {e}")
            raise

    def _fetch_spot_price_data(self, commodity_code: str, assessment_date: date) -> Dict[str, Any]:
        """Fetch spot price data for a specific European hub."""

        # Mock spot price data for development
        # In production: Query exchange APIs (ICE, EEX, etc.)

        # European hub price relationships (TTF as benchmark)
        hub_premiums = {
            "TTF": 0.0,       # Benchmark
            "NBP": -2.0,      # UK discount due to transport costs
            "CEGH": 1.0,      # Austrian premium for pipeline access
            "PSV": 2.0,       # Italian premium for LNG imports
            "PEG": 0.5        # French slight premium
        }

        base_price = 25.0  # €25/MWh TTF equivalent
        premium = hub_premiums.get(commodity_code, 0.0)

        # Add realistic daily variation
        import random
        price_variation = random.uniform(-1.0, 1.0)  # +/- €1/MWh
        final_price = base_price + premium + price_variation

        # Convert to appropriate units
        if commodity_code == "NBP":
            # Convert €/MWh to pence/therm (rough conversion)
            final_price = final_price * 0.35 * 100  # Convert to pence/therm

        return {
            "price": round(final_price, 2),
            "volume": random.randint(50000, 200000),  # Typical daily volumes
            "high": round(final_price + random.uniform(0.5, 1.5), 2),
            "low": round(final_price - random.uniform(0.5, 1.5), 2),
            "change": round(price_variation, 2)
        }

    def _get_unit_for_hub(self, commodity_code: str) -> str:
        """Get the appropriate unit for each hub."""
        units = {
            "TTF": "EUR/MWh",
            "NBP": "GBP/therm",
            "CEGH": "EUR/MWh",
            "PSV": "EUR/MWh",
            "PEG": "EUR/MWh"
        }
        return units.get(commodity_code, "EUR/MWh")

    def _fetch_forward_curve_data(self, commodity_code: str, assessment_date: date) -> List[Dict[str, Any]]:
        """Fetch forward curve data for European hubs."""
        # Mock forward curve data
        # In production: Query exchange APIs for forward curves

        base_prices = {
            "TTF": 25.0,
            "NBP": 22.0,
            "CEGH": 26.0,
            "PSV": 27.0,
            "PEG": 25.5
        }

        base_price = base_prices.get(commodity_code, 25.0)

        forward_contracts = []
        for i in range(1, 13):  # Next 12 months
            contract_month = assessment_date.replace(month=assessment_date.month + i)
            if contract_month.month > 12:
                contract_month = contract_month.replace(year=contract_month.year + 1, month=contract_month.month - 12)

            # European forward curves typically show summer discount, winter premium
            months_ahead = i
            seasonal_factor = 1.0

            if contract_month.month in [6, 7, 8]:  # Summer
                seasonal_factor = 0.9  # 10% summer discount
            elif contract_month.month in [12, 1, 2]:  # Winter
                seasonal_factor = 1.1  # 10% winter premium

            price = base_price * seasonal_factor * (1 + months_ahead * 0.01)  # Slight contango

            forward_contracts.append({
                "contract_month": contract_month,
                "settlement_price": round(price, 2),
                "open_interest": int(100000 * (1 / months_ahead)),
                "volume": int(50000 * (1 / months_ahead))
            })

        return forward_contracts
