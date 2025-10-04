"""Argus Media oil benchmark connector (stub with mocks)."""

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
        # Base class requires implementation; handled via _register_specs
        pass

    def _register_specs(self) -> None:
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


