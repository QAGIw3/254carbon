"""
Data models for 254Carbon SDK.
"""
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Instrument(BaseModel):
    """Market instrument."""
    instrument_id: str
    market: str
    product: str
    location_code: str
    timezone: str = "UTC"
    unit: str
    currency: str = "USD"


class PriceTick(BaseModel):
    """Price tick event."""
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
    """Forward curve point."""
    delivery_start: date
    delivery_end: date
    tenor_type: str
    price: float
    currency: str = "USD"
    unit: str


class ForwardCurve(BaseModel):
    """Forward curve."""
    instrument_id: str
    as_of_date: date
    scenario_id: str = "BASE"
    points: List[Dict[str, Any]]


class Scenario(BaseModel):
    """Scenario definition."""
    scenario_id: str
    title: str
    description: str
    visibility: str = "org"
    created_by: str
    created_at: datetime


class ScenarioRun(BaseModel):
    """Scenario execution run."""
    run_id: str
    scenario_id: str
    status: str  # queued, running, success, failed
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    notes: Optional[str] = None

