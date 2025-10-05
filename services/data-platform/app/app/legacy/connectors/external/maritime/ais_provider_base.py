"""Base interfaces and models for AIS providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional


@dataclass(slots=True)
class VesselPosition:
    imo: Optional[str]
    mmsi: str
    latitude: float
    longitude: float
    speed_knots: float
    course_deg: Optional[float]
    heading_deg: Optional[float]
    timestamp: datetime
    status: Optional[str]
    destination: Optional[str]


@dataclass(slots=True)
class PortCallEvent:
    imo: Optional[str]
    mmsi: str
    terminal_id: str
    event_type: str  # arrival|departure|berth|anchor
    event_time: datetime
    status: Optional[str]
    metadata: Dict[str, str]


class AISProvider(abc.ABC):
    """Abstract base class describing AIS provider capabilities."""

    @abc.abstractmethod
    def fetch_positions(self, area: Dict[str, float]) -> Iterable[VesselPosition]:
        """Fetch vessel positions within a bounding box.

        area: {"min_lat": ..., "max_lat": ..., "min_lon": ..., "max_lon": ...}
        """

    @abc.abstractmethod
    def fetch_port_calls(
        self,
        terminal_ids: List[str],
        lookback_hours: int,
    ) -> Iterable[PortCallEvent]:
        """Fetch port call events for specified terminals within lookback window."""
