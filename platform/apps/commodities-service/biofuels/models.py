"""
Biofuels schemas.

Extends the shared `PricePoint` to annotate RIN-specific metadata, e.g.
rin_type, while harmonizing fields across endpoints for simpler clients.
"""
from typing import Optional

from schemas.pricing import PricePoint


class RINPricePoint(PricePoint):
    rin_type: Optional[str] = None


__all__ = ["RINPricePoint"]
