"""
Data models for 254Carbon SDK.

These Pydantic models provide type‑safe structures for common API resources used
throughout the SDK. They are intentionally minimal and map closely to the API
responses to keep the client lightweight.
"""
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Instrument(BaseModel):
    """Market instrument.

    Attributes
    ----------
    instrument_id: Canonical identifier (e.g., "MISO.HUB.INDIANA").
    market: Market category (e.g., power, gas).
    product: Product type (e.g., lmp, curve).
    location_code: Regional identifier or node code.
    timezone: Olson TZ name for temporal data alignment.
    unit: Quoted unit (e.g., MWh, MMBtu).
    currency: Currency code for monetary values.
    """
    instrument_id: str
    market: str
    product: str
    location_code: str
    timezone: str = "UTC"
    unit: str
    currency: str = "USD"


class PriceTick(BaseModel):
    """Price tick event.

    Each instance represents a single observation across a market instrument at
    a given time with optional volume and source metadata.
    """
    event_time: datetime
    instrument_id: str
    location_code: str
    price_type: str
    value: float
    volume: Optional[float] = None
    currency: str = "USD"
    unit: str
    source: str


class CurvePoint(BaseModel):
    """Forward curve point covering a delivery period."""
    delivery_start: date
    delivery_end: date
    tenor_type: str
    price: float
    currency: str = "USD"
    unit: str


class ForwardCurve(BaseModel):
    """Forward curve container.

    Notes
    -----
    ``points`` holds a list of dicts for flexibility across tenor schemas. Use
    ``CurvePoint`` when a stricter schema is required downstream.
    """
    instrument_id: str
    as_of_date: date
    scenario_id: str = "BASE"
    points: List[Dict[str, Any]]


class Scenario(BaseModel):
    """Scenario definition with metadata."""
    scenario_id: str
    title: str
    description: str
    visibility: str = "org"
    created_by: str
    created_at: datetime


class ScenarioRun(BaseModel):
    """Scenario execution run and status markers."""
    run_id: str
    scenario_id: str
    status: str  # queued, running, success, failed
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    notes: Optional[str] = None
