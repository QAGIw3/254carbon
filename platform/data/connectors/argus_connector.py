"""
Argus Oil Benchmarks Connector

Overview
--------
Lightweight scaffold for ingesting Argus Media physical oil benchmarks and
assessments. This implementation intentionally uses mocked data to keep
integration predictable in development environments while documenting the
intended data flow and contract metadata wiring.

Data Flow
---------
Argus source (API/file) → normalize assessment → canonical price event → Kafka

Configuration
-------------
- `kafka.topic`: Destination topic for canonical ticks (defaults to
  `commodities.ticks.v1`).
- `market`: Optional source/market label (defaults to `ARGUS`).

Operational Notes
-----------------
- This is a stub that emits a small set of representative instruments. Swap
  `pull_or_subscribe` with real API/file reads when credentials and format are
  available.
- `map_to_schema` mirrors the canonical schema fields but keeps types minimal
  for the mock. When moving to production, ensure timestamp types and units
  conform to platform standards.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, date
from typing import Dict, Any, Iterator

from .commodities.base import BaseCommodityConnector
from .base import CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class ArgusOilConnector(BaseCommodityConnector):
    commodity_type = CommodityType.OIL

    def __init__(self, config: Dict[str, Any]):
        cfg = dict(config)
        cfg.setdefault("kafka", {}).setdefault("topic", "commodities.ticks.v1")
        super().__init__(cfg)
        self.market = cfg.get("market", "ARGUS")
        self._register_specs()

    def register_contracts(self) -> None:
        """Required by base; specs are registered in `_register_specs`.

        Some connectors build specifications dynamically. Here we centralize in a
        private helper to keep constructor tidy.
        """
        pass

    def _register_specs(self) -> None:
        """Register a minimal set of Argus physical benchmarks.

        Notes
        -----
        The attributes here are illustrative. In a production connector, you
        would import official contract metadata (codes, quality specs, delivery
        locations) from a vetted catalog rather than hard-coding.
        """
        for code, location in [
            ("ARGUS_WTI_HOUSTON", "Houston, TX"),
            ("ARGUS_MARS", "Gulf of Mexico"),
            ("ARGUS_WTI_MIDLAND", "Midland, TX"),
        ]:
            self.register_contract_specification(
                ContractSpecification(
                    commodity_code=code,
                    commodity_type=CommodityType.OIL,
                    contract_unit="barrel",
                    quality_spec={
                        "notes": "Argus assessment",
                    },
                    delivery_location=location,
                    exchange="ARGUS",
                    listing_rules={"roll_window_days": 5},
                )
            )

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Yield mocked Argus price assessments for a few key grades.

        Replace this stub with real Argus API/file ingestion. For streaming
        sources, this method can subscribe to a feed and yield indefinitely; for
        batch sources, it should yield a bounded set each run based on
        checkpoint state.
        """
        # Mock data for development
        now = datetime.now(timezone.utc)
        for idx, code in enumerate(self.contract_specs.keys()):
            price = 80 + idx * 1.5
            yield self.create_price_event(
                commodity_code=code,
                price=price,
                event_time=now,
                price_type="assessment",
            )

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map internal event dict to the canonical wire format.

        Important: This stub preserves `event_time_utc` as an ISO string to
        match light-weight mocks elsewhere in the dev harness. Production code
        should emit epoch milliseconds and include `version_id` as needed by the
        downstream schema contracts.
        """
        # Pass-through mapping of stabilized fields. Keep this aligned with
        # `create_price_event` from the base classes for consistency.
        return {
            "event_time_utc": raw["event_time"].isoformat(),
            "arrival_time_utc": datetime.now(timezone.utc).isoformat(),
            "market": raw["commodity_type"],
            "product": raw["commodity_type"],
            "instrument_id": raw["instrument_id"],
            "location_code": raw["location_code"],
            "price_type": raw["price_type"],
            "value": raw["value"],
            "volume": raw.get("volume"),
            "currency": raw["currency"],
            "unit": raw["unit"],
            "source": raw["source"],
            "commodity_type": raw["commodity_type"],
        }

