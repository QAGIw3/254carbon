"""EIA petroleum connector for crude benchmarks and regional assessments."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, date
from typing import Dict, Any, Iterator, Optional, List

import requests

from .commodities.base import BaseCommodityConnector
from .base import CommodityType, ContractSpecification

logger = logging.getLogger(__name__)


class EIAPetroleumConnector(BaseCommodityConnector):
    commodity_type = CommodityType.OIL

    def __init__(self, config: Dict[str, Any]):
        cfg = dict(config)
        super().__init__(cfg)
        self.api_base = cfg.get("api_base", "https://api.eia.gov/v2/petroleum/pri/data/")
        self.api_key = cfg.get("api_key")
        if not self.api_key:
            raise ValueError("EIA API key required")
        self.symbols = cfg.get(
            "series_codes",
            {
                "WTI_CUSHING": "PET.RWTC.D",
                "BRENT": "PET.RBRTE.D",
            },
        )
        self.register_contracts()

    def register_contracts(self) -> None:
        self.register_contract_specification(
            ContractSpecification(
                commodity_code="WTI_CUSHING",
                commodity_type=CommodityType.OIL,
                contract_unit="barrel",
                quality_spec={
                    "api_gravity": 39.6,
                    "sulfur_pct": 0.24,
                },
                delivery_location="Cushing, OK",
                exchange="EIA",
                listing_rules={"roll_window_days": 5},
            )
        )
        self.register_contract_specification(
            ContractSpecification(
                commodity_code="BRENT",
                commodity_type=CommodityType.OIL,
                contract_unit="barrel",
                quality_spec={
                    "api_gravity": 38.3,
                    "sulfur_pct": 0.37,
                },
                delivery_location="North Sea",
                exchange="EIA",
                listing_rules={"roll_window_days": 5},
            )
        )

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date
        for code, series_id in self.symbols.items():
            params = {
                "api_key": self.api_key,
                "frequency": "daily",
                "data[0]": "value",
                "facets[item]": [],
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "start": 0,
                "length": 1,
            }
            url = f"{self.api_base}?series_id={series_id}"
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("response", {}).get("data", [])
            for row in rows:
                price = float(row["value"])
                period = datetime.strptime(row["period"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                yield self.create_price_event(
                    commodity_code=code,
                    price=price,
                    event_time=period,
                    price_type="spot",
                    unit="USD/bbl",
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
            "value": float(raw["value"]),
            "volume": raw.get("volume"),
            "currency": raw["currency"],
            "unit": raw["unit"],
            "source": raw["source"],
            "commodity_type": raw["commodity_type"],
        }


