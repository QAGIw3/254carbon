"""
Commodity Connector Base

Purpose
-------
Provide shared behavior for commodity-focused connectors (oil, gas, coal,
etc.). This layer sits on top of the generic `Ingestor` to add convenient
helpers for expressing price ticks, futures curves, and simple rollover
evaluation without re-implementing boilerplate in every connector.

Design
------
- Contract metadata is expressed via `ContractSpecification` and registered on
  the connector. Helpers use this registry to populate canonical fields.
- Event creation helpers (`create_price_event`, `emit_futures_curve_events`)
  ensure consistency with the platform’s wire format and reduce drift across
  connectors.

Notes
-----
Emitters here defer to `Ingestor.emit`, which handles batching, Kafka write
settings, and serialization.
"""

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
    """Shared behaviour for commodity-specific connectors.

    Subclasses should set `commodity_type` and implement `register_contracts`
    to populate the internal contract registry used by helper methods.
    """

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
        """Subclasses should register `ContractSpecification` entries.

        Implementations typically call `register_contract_specification` one or
        more times during initialization.
        """

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
        """Build a canonical price event from minimal inputs.

        Parameters mirror the common case for assessed or traded prices while
        reusing the contract registry to fill in currency/unit defaults when
        omitted.
        """
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
        """Emit a batch of price events with base retry/flush semantics."""
        return self.emit(events)

    def emit_futures_curve_events(
        self,
        commodity_code: str,
        as_of_date: date,
        contracts: Iterable[FuturesContract],
    ) -> int:
        """Emit a curve by mapping `FuturesContract` instances to events."""
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
        """Apply generic rollover rules to the front of the curve.

        Returns a tuple `(active, deferred)` which callers can use to update
        positions or compute front/deferred spreads.
        """
        if len(contracts) < 2:
            raise ValueError("Need at least two contracts to evaluate rollover")
        current_contract, next_contract = contracts[0], contracts[1]
        return self.handle_contract_rollover(
            commodity_code=commodity_code,
            current_contract=current_contract,
            next_contract=next_contract,
            thresholds=thresholds,
        )

