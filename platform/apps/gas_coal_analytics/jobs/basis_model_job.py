"""Scheduled job for gas basis modeling."""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, List

from ..analytics.gas_basis import GasBasisModeler
from ..models import GasBasisModelResult

logger = logging.getLogger(__name__)


def run_basis_model_job(as_of: date, hubs: Iterable[str]) -> List[GasBasisModelResult]:
    modeler = GasBasisModeler()
    outputs: List[GasBasisModelResult] = []
    for hub in hubs:
        try:
            result = modeler.compute(hub, as_of)
            modeler.persist(result)
            outputs.append(result)
            logger.info("Persisted gas basis model output for %s on %s", hub, as_of)
        except Exception as exc:
            logger.exception("Basis model job failed for %s: %s", hub, exc)
    return outputs
