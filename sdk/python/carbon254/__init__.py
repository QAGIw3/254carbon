"""
254Carbon Python SDK

Official Python client library for the 254Carbon Market Intelligence Platform.
"""
from .client import CarbonClient
from .models import (
    Instrument,
    PriceTick,
    ForwardCurve,
    Scenario,
    ScenarioRun,
)
from .exceptions import (
    CarbonAPIError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
)

__version__ = "1.0.0"
__all__ = [
    "CarbonClient",
    "Instrument",
    "PriceTick",
    "ForwardCurve",
    "Scenario",
    "ScenarioRun",
    "CarbonAPIError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
]

