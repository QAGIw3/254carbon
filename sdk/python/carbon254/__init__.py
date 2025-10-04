"""
254Carbon Python SDK

Overview
--------
Official Python client library for the 254Carbon Market Intelligence Platform.
This package provides a typed client, models, and helpers for synchronous,
asynchronous, and streaming use cases. It is designed to be easy to adopt for
both exploratory analysis (via pandas) and production services (via httpx).

Exports
-------
- ``CarbonClient``: main entry point for API access
- Pydantic models for typed responses: ``Instrument``, ``PriceTick``,
  ``ForwardCurve``, ``Scenario``, ``ScenarioRun``
- Exception hierarchy rooted at ``CarbonAPIError``
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

# Package semantic version. Keep in sync with packaging config in setup.py
__version__ = "1.0.0"
# Public API surface intended for ``from carbon254 import *`` consumers.
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
