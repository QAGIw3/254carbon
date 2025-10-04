"""Scheduled job for weather impact analytics."""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, List

from ..analytics.weather_impact import WeatherImpactAnalyzer
from ..models import WeatherImpactCoefficient

logger = logging.getLogger(__name__)


def run_weather_impact_job(as_of: date, entities: Iterable[str], window: int = 120) -> List[WeatherImpactCoefficient]:
    analyzer = WeatherImpactAnalyzer(window=window)
    outputs: List[WeatherImpactCoefficient] = []
    for entity in entities:
        try:
            coeffs = analyzer.run(entity, as_of)
            analyzer.persist(coeffs)
            outputs.extend(coeffs)
            logger.info("Weather impact coefficients computed for %s (%d entries)", entity, len(coeffs))
        except Exception as exc:
            logger.exception("Weather impact job failed for %s: %s", entity, exc)
    return outputs
