"""Scheduled job for storage arbitrage analytics."""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, List

from ..analytics.storage_arbitrage import StorageArbitrageCalculator
from ..models import StorageArbitrageResult

logger = logging.getLogger(__name__)


def run_storage_arbitrage_job(as_of: date, hubs: Iterable[str]) -> List[StorageArbitrageResult]:
    calculator = StorageArbitrageCalculator()
    results: List[StorageArbitrageResult] = []
    for hub in hubs:
        try:
            result = calculator.compute(hub, as_of)
            calculator.persist(result)
            results.append(result)
            logger.info("Persisted storage arbitrage result for %s on %s", hub, as_of)
        except Exception as exc:
            logger.exception("Storage arbitrage job failed for %s: %s", hub, exc)
    return results
