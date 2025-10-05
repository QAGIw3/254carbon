"""LNG vessel tracking connector.

Aggregates AIS data to compute vessel counts and port call metrics for LNG
terminals. The connector depends on a provider implementing `AISProvider`
interface and emits fundamentals events.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ...base import CommodityType, Ingestor
from .ais_provider_base import AISProvider, PortCallEvent, VesselPosition

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LNGTerminal:
    terminal_id: str
    name: str
    country: str
    region: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


class LNGVesselTrackingConnector(Ingestor):
    """Connector aggregating LNG vessel activity metrics."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.GAS

        self.provider: AISProvider = config["provider"]
        self.terminals: List[LNGTerminal] = [
            LNGTerminal(**terminal)
            if isinstance(terminal, dict)
            else terminal
            for terminal in config.get("terminals", [])
        ]
        self.lookback_hours = int(config.get("lookback_hours", 24))
        self.kafka_topic = config.get("kafka", {}).get("topic", "market.fundamentals")

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "terminals": [terminal.terminal_id for terminal in self.terminals],
            "metrics": [
                "lng_vessels_at_berth_count",
                "lng_vessels_waiting_count",
                "lng_arrivals_24h_count",
                "lng_departures_24h_count",
            ],
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        lookback_start = now - timedelta(hours=self.lookback_hours)

        terminal_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "berth": 0,
                "waiting": 0,
                "arrivals": 0,
                "departures": 0,
                "vessels": set(),
            }
        )

        for terminal in self.terminals:
            area = {
                "min_lat": terminal.min_lat,
                "max_lat": terminal.max_lat,
                "min_lon": terminal.min_lon,
                "max_lon": terminal.max_lon,
            }
            try:
                positions = list(self.provider.fetch_positions(area))
            except Exception as exc:
                logger.error("Failed to fetch positions for %s: %s", terminal.terminal_id, exc)
                continue

            for pos in positions:
                key = terminal.terminal_id
                terminal_metrics[key]["vessels"].add(pos.mmsi)
                if pos.status and pos.status.lower() in {"moored", "berthed"}:
                    terminal_metrics[key]["berth"] += 1
                else:
                    terminal_metrics[key]["waiting"] += 1

        try:
            port_calls = list(self.provider.fetch_port_calls(
                [t.terminal_id for t in self.terminals], self.lookback_hours
            ))
        except Exception as exc:
            logger.error("Failed to fetch port calls: %s", exc)
            port_calls = []

        for call in port_calls:
            metrics = terminal_metrics.get(call.terminal_id)
            if not metrics:
                continue
            if call.event_type == "arrival" and call.event_time >= lookback_start:
                metrics["arrivals"] += 1
            if call.event_type == "departure" and call.event_time >= lookback_start:
                metrics["departures"] += 1

        for terminal in self.terminals:
            metrics = terminal_metrics[terminal.terminal_id]
            yield from self._metrics_to_events(terminal, metrics, now)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts: datetime = raw["event_time"]
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "infra",
            "product": raw["product"],
            "instrument_id": raw["instrument_id"],
            "location_code": raw["location_code"],
            "price_type": "observation",
            "value": raw["value"],
            "unit": raw["unit"],
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "metadata": json.dumps(raw.get("metadata", {})),
        }

    def emit(self, events: Iterable[Dict[str, Any]]) -> int:
        return super().emit(events)

    def checkpoint(self, state: Dict[str, Any]) -> None:
        super().checkpoint(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _metrics_to_events(
        self,
        terminal: LNGTerminal,
        metrics: Dict[str, Any],
        event_time: datetime,
    ) -> Iterator[Dict[str, Any]]:
        instrument_id = f"LNG.{terminal.terminal_id}.VESSELS"
        base_metadata = {
            "terminal_name": terminal.name,
            "country": terminal.country,
            "region": terminal.region,
            "lookback_hours": self.lookback_hours,
            "vessels_sampled": len(metrics.get("vessels", [])),
        }

        yield {
            "event_time": event_time,
            "product": "lng_vessels_at_berth_count",
            "value": metrics["berth"],
            "unit": "count",
            "instrument_id": instrument_id,
            "location_code": terminal.terminal_id,
            "metadata": base_metadata,
        }

        yield {
            "event_time": event_time,
            "product": "lng_vessels_waiting_count",
            "value": metrics["waiting"],
            "unit": "count",
            "instrument_id": instrument_id,
            "location_code": terminal.terminal_id,
            "metadata": base_metadata,
        }

        yield {
            "event_time": event_time,
            "product": "lng_arrivals_24h_count",
            "value": metrics["arrivals"],
            "unit": "count",
            "instrument_id": instrument_id,
            "location_code": terminal.terminal_id,
            "metadata": base_metadata,
        }

        yield {
            "event_time": event_time,
            "product": "lng_departures_24h_count",
            "value": metrics["departures"],
            "unit": "count",
            "instrument_id": instrument_id,
            "location_code": terminal.terminal_id,
            "metadata": base_metadata,
        }
