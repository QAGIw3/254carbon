"""
European Power Markets Connector

Ingests data from major European power exchanges:
- EPEX Spot (France, Germany, Netherlands, Belgium, Austria, Switzerland)
- Nord Pool (Nordic and Baltic countries)
- Poland Power Exchange (TGE)
- Eastern European markets

Supports day-ahead, intraday, and balancing markets.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator, Dict, Any, Optional, List
import time

import requests
from kafka import KafkaProducer
import json

from .base import Ingestor

logger = logging.getLogger(__name__)


class EuropeanMarketsConnector(Ingestor):
    """
    European power markets data connector.

    Responsibilities
    - Ingest day-ahead prices from EPEX Spot and Nord Pool
    - Ingest intraday prices and volumes
    - Map European market structures to canonical schema
    - Handle multiple timezones and currencies (EUR)
    - Provide fallback to mock data for development

    Production notes
    - EPEX: https://api.epexspot.com/ (requires API key)
    - Nord Pool: https://www.nordpoolgroup.com/api/ (public data)
    - TGE: https://tge.pl/ (Polish Power Exchange)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.market = config.get("market", "EPEX")  # EPEX, NORDPOOL, TGE
        self.api_base = config.get("api_base", "https://api.epexspot.com")
        self.api_key = config.get("api_key")
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")

        # European timezone mappings
        self.timezone_map = {
            "DE": "Europe/Berlin",
            "FR": "Europe/Paris",
            "NL": "Europe/Amsterdam",
            "BE": "Europe/Brussels",
            "AT": "Europe/Vienna",
            "CH": "Europe/Zurich",
            "NO": "Europe/Oslo",
            "SE": "Europe/Stockholm",
            "DK": "Europe/Copenhagen",
            "FI": "Europe/Helsinki",
            "PL": "Europe/Warsaw",
            "CZ": "Europe/Prague",
            "HU": "Europe/Budapest",
            "RO": "Europe/Bucharest"
        }

        self.producer = None

    def discover(self) -> Dict[str, Any]:
        """Discover European market data streams."""
        return {
            "source_id": self.source_id,
            "markets": [
                {
                    "name": "EPEX_SPOT",
                    "description": "European Power Exchange Spot Market",
                    "regions": ["DE", "FR", "NL", "BE", "AT", "CH"],
                    "products": ["day_ahead", "intraday"],
                    "frequency": "hourly",
                    "currency": "EUR"
                },
                {
                    "name": "NORD_POOL",
                    "description": "Nordic and Baltic Power Market",
                    "regions": ["NO", "SE", "DK", "FI", "EE", "LV", "LT"],
                    "products": ["day_ahead", "intraday", "balancing"],
                    "frequency": "hourly",
                    "currency": "EUR"
                },
                {
                    "name": "TGE",
                    "description": "Polish Power Exchange",
                    "regions": ["PL"],
                    "products": ["day_ahead", "intraday"],
                    "frequency": "hourly",
                    "currency": "PLN"
                }
            ]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull European market data.

        In production, this would call actual exchange APIs.
        For development, provides realistic mock data.
        """
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.now(timezone.utc) - timedelta(hours=24)
        )

        logger.info(f"Fetching European market data since {last_time}")

        if self.market == "EPEX":
            yield from self._fetch_epex_data()
        elif self.market == "NORDPOOL":
            yield from self._fetch_nordpool_data()
        elif self.market == "TGE":
            yield from self._fetch_tge_data()
        else:
            logger.error(f"Unsupported European market: {self.market}")
            return

    def _fetch_epex_data(self) -> Iterator[Dict[str, Any]]:
        """Fetch EPEX Spot market data."""
        try:
            # EPEX API endpoints (would need actual API integration)
            # For now, generate realistic mock data

            regions = ["DE", "FR", "NL", "BE", "AT", "CH"]
            current_date = datetime.now(timezone.utc).date()

            for region in regions:
                for hour in range(24):
                    # Create timestamp for this hour
                    timestamp = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour)
                    timestamp = timestamp.replace(tzinfo=timezone.utc)

                    # Generate realistic EPEX prices
                    base_price = 45.0  # EUR/MWh base price

                    # Regional price variations
                    region_multipliers = {
                        "DE": 1.0,    # Reference price
                        "FR": 1.05,   # Slightly higher
                        "NL": 0.95,   # Slightly lower
                        "BE": 1.02,   # Moderate
                        "AT": 0.98,   # Lower
                        "CH": 1.08    # Higher due to hydro
                    }

                    price = base_price * region_multipliers.get(region, 1.0)

                    # Add time-of-day variations
                    if 8 <= hour <= 11 or 17 <= hour <= 20:  # Peak hours
                        price *= 1.3
                    elif 0 <= hour <= 6:  # Off-peak
                        price *= 0.7

                    # Add some randomness
                    import random
                    price *= (1 + random.uniform(-0.1, 0.1))

                    # Map to canonical format
                    instrument_id = f"EPEX.HUB.{region}"

                    yield {
                        "event_time": timestamp,
                        "instrument_id": instrument_id,
                        "market": "power",
                        "product": "day_ahead",
                        "price": round(price, 2),
                        "volume": 1000.0 + random.uniform(500, 2000),
                        "currency": "EUR",
                        "unit": "MWh",
                        "source": "epex",
                        "region": region,
                        "metadata": {
                            "delivery_hour": hour,
                            "delivery_date": current_date.isoformat(),
                            "timezone": self.timezone_map.get(region, "Europe/Berlin")
                        }
                    }

        except Exception as e:
            logger.error(f"Error fetching EPEX data: {e}")
            # Fallback to minimal mock data
            yield from self._get_minimal_epex_mock()

    def _fetch_nordpool_data(self) -> Iterator[Dict[str, Any]]:
        """Fetch Nord Pool market data."""
        try:
            regions = ["NO", "SE", "DK", "FI"]
            current_date = datetime.now(timezone.utc).date()

            for region in regions:
                for hour in range(24):
                    timestamp = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour)
                    timestamp = timestamp.replace(tzinfo=timezone.utc)

                    # Nordic prices tend to be lower due to hydro
                    base_price = 35.0  # EUR/MWh

                    # Regional variations
                    region_multipliers = {
                        "NO": 0.9,    # Low due to hydro
                        "SE": 0.95,   # Moderate
                        "DK": 1.0,    # Average
                        "FI": 0.85    # Low nuclear/hydro mix
                    }

                    price = base_price * region_multipliers.get(region, 1.0)

                    # Seasonal variations
                    month = timestamp.month
                    if month in [12, 1, 2]:  # Winter
                        price *= 1.2
                    elif month in [6, 7, 8]:  # Summer
                        price *= 0.8

                    import random
                    price *= (1 + random.uniform(-0.15, 0.15))

                    instrument_id = f"NORDPOOL.HUB.{region}"

                    yield {
                        "event_time": timestamp,
                        "instrument_id": instrument_id,
                        "market": "power",
                        "product": "day_ahead",
                        "price": round(price, 2),
                        "volume": 800.0 + random.uniform(400, 1500),
                        "currency": "EUR",
                        "unit": "MWh",
                        "source": "nordpool",
                        "region": region,
                        "metadata": {
                            "delivery_hour": hour,
                            "delivery_date": current_date.isoformat(),
                            "timezone": self.timezone_map.get(region, "Europe/Oslo")
                        }
                    }

        except Exception as e:
            logger.error(f"Error fetching Nord Pool data: {e}")
            yield from self._get_minimal_nordpool_mock()

    def _fetch_tge_data(self) -> Iterator[Dict[str, Any]]:
        """Fetch Polish Power Exchange (TGE) data."""
        try:
            current_date = datetime.now(timezone.utc).date()

            for hour in range(24):
                timestamp = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour)
                timestamp = timestamp.replace(tzinfo=timezone.utc)

                # Polish prices typically higher than Nordic but lower than continental Europe
                base_price = 50.0  # EUR/MWh

                # Add time-of-day variations
                if 7 <= hour <= 10 or 16 <= hour <= 19:  # Peak hours
                    base_price *= 1.25
                elif 1 <= hour <= 6:  # Night
                    base_price *= 0.75

                import random
                price = base_price * (1 + random.uniform(-0.1, 0.1))

                instrument_id = "TGE.HUB.PL"

                yield {
                    "event_time": timestamp,
                    "instrument_id": instrument_id,
                    "market": "power",
                    "product": "day_ahead",
                    "price": round(price, 2),
                    "volume": 600.0 + random.uniform(300, 1200),
                    "currency": "PLN",
                    "unit": "MWh",
                    "source": "tge",
                    "region": "PL",
                    "metadata": {
                        "delivery_hour": hour,
                        "delivery_date": current_date.isoformat(),
                        "timezone": "Europe/Warsaw"
                    }
                }

        except Exception as e:
            logger.error(f"Error fetching TGE data: {e}")
            yield from self._get_minimal_tge_mock()

    def _get_minimal_epex_mock(self) -> Iterator[Dict[str, Any]]:
        """Minimal mock data for EPEX."""
        timestamp = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        for region in ["DE", "FR"]:
            yield {
                "event_time": timestamp,
                "instrument_id": f"EPEX.HUB.{region}",
                "market": "power",
                "product": "day_ahead",
                "price": 45.0,
                "volume": 1000.0,
                "currency": "EUR",
                "unit": "MWh",
                "source": "epex",
                "region": region
            }

    def _get_minimal_nordpool_mock(self) -> Iterator[Dict[str, Any]]:
        """Minimal mock data for Nord Pool."""
        timestamp = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        for region in ["NO", "SE"]:
            yield {
                "event_time": timestamp,
                "instrument_id": f"NORDPOOL.HUB.{region}",
                "market": "power",
                "product": "day_ahead",
                "price": 35.0,
                "volume": 800.0,
                "currency": "EUR",
                "unit": "MWh",
                "source": "nordpool",
                "region": region
            }

    def _get_minimal_tge_mock(self) -> Iterator[Dict[str, Any]]:
        """Minimal mock data for TGE."""
        timestamp = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        yield {
            "event_time": timestamp,
            "instrument_id": "TGE.HUB.PL",
            "market": "power",
            "product": "day_ahead",
            "price": 50.0,
            "volume": 600.0,
            "currency": "PLN",
            "unit": "MWh",
            "source": "tge",
            "region": "PL"
        }

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map European market data to canonical schema."""
        return {
            "event_time": raw["event_time"],
            "instrument_id": raw["instrument_id"],
            "market": raw["market"],
            "product": raw["product"],
            "price": raw["price"],
            "volume": raw.get("volume"),
            "currency": raw["currency"],
            "unit": raw["unit"],
            "source": raw["source"],
            "region": raw["region"],
            "metadata": raw.get("metadata", {})
        }

    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """Emit events to Kafka."""
        if not self.producer:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                retry_backoff_ms=1000
            )

        emitted_count = 0

        for event in events:
            try:
                # Use instrument_id as partition key for consistent routing
                key = event["instrument_id"]

                # Send to Kafka
                future = self.producer.send(
                    self.kafka_topic,
                    key=key,
                    value=event
                )

                # Wait for acknowledgment in production
                if self.config.get("wait_for_ack", False):
                    future.get(timeout=10)

                emitted_count += 1

            except Exception as e:
                logger.error(f"Error emitting event to Kafka: {e}")
                continue

        return emitted_count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state."""
        super().checkpoint(state)

        # Update last processed timestamp
        if "last_event_time" not in state:
            state["last_event_time"] = datetime.now(timezone.utc).isoformat()

        # Save market-specific state
        state["last_market_update"] = {
            "market": self.market,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
