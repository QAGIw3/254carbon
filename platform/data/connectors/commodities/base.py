"""Commodity connector abstractions and utilities."""

from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import datetime, timezone, date
from typing import Dict, Any, Iterable, Optional, Tuple, List

from ..base import (
    Ingestor,
    CommodityType,
    ContractSpecification,
    FuturesContract,
)

logger = logging.getLogger(__name__)


class BaseCommodityConnector(Ingestor):
    """Shared behaviour for commodity-specific connectors."""

    commodity_type: CommodityType

    def __init__(self, config: Dict[str, Any]):
        cfg = dict(config)
        kafka_cfg = cfg.setdefault("kafka", {})
        kafka_cfg.setdefault("topic", "commodities.ticks.v1")
        super().__init__(cfg)
        if not hasattr(self, "commodity_type"):
            raise ValueError("commodity_type must be defined by subclasses")

    @abstractmethod
    def register_contracts(self) -> None:
        """Subclasses should register ContractSpecification entries."""

    def discover(self) -> Dict[str, Any]:
        specs = [spec.to_dict() for spec in self.contract_specs.values()]
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "contracts": specs,
        }

    def create_price_event(
        self,
        commodity_code: str,
        price: float,
        event_time: datetime,
        price_type: str = "spot",
        volume: Optional[float] = None,
        location_code: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.create_commodity_price_event(
            commodity_code=commodity_code,
            price=price,
            event_time=event_time,
            price_type=price_type,
            volume=volume,
            location_code=location_code,
            unit=unit or "USD/bbl",
        )

    def emit_price_events(self, events: Iterable[Dict[str, Any]]) -> int:
        return self.emit(events)

    def emit_futures_curve_events(
        self,
        commodity_code: str,
        as_of_date: date,
        contracts: Iterable[FuturesContract],
    ) -> int:
        events = (
            self.create_futures_curve_event(
                commodity_code=commodity_code,
                as_of_date=as_of_date,
                contract_data={
                    "contract_month": contract.contract_month,
                    "settlement_price": contract.settlement_price,
                    "open_interest": contract.open_interest,
                    "volume": contract.volume,
                },
            )
            for contract in contracts
        )
        return self.emit(events)

    def evaluate_rollover(
        self,
        commodity_code: str,
        contracts: List[FuturesContract],
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> Tuple[FuturesContract, FuturesContract]:
        if len(contracts) < 2:
            raise ValueError("Need at least two contracts to evaluate rollover")
        current_contract, next_contract = contracts[0], contracts[1]
        return self.handle_contract_rollover(
            commodity_code=commodity_code,
            current_contract=current_contract,
            next_contract=next_contract,
            thresholds=thresholds,
        )


