"""Scheduled job for coal-to-gas switching analytics."""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, List, Optional

from ..analytics.coal_to_gas import CoalToGasSwitchingCalculator
from ..models import CoalToGasSwitchingResult

logger = logging.getLogger(__name__)


def run_coal_to_gas_switch_job(
    as_of: date,
    regions: Iterable[str],
    co2_price: Optional[float] = None,
) -> List[CoalToGasSwitchingResult]:
    calculator = CoalToGasSwitchingCalculator()
    results: List[CoalToGasSwitchingResult] = []
    for region in regions:
        try:
            result = calculator.compute(region, as_of, co2_price=co2_price)
            calculator.persist(result)
            results.append(result)
            logger.info(
                "Persisted coal-to-gas switching metrics for %s on %s", region, as_of
            )
        except Exception as exc:
            logger.exception("Coal-to-gas switching job failed for %s: %s", region, exc)
    return results
