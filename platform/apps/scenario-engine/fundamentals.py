"""Fundamentals layer for scenario engine."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class FundamentalsResult:
    load_forecast: Dict[str, Any]
    generation_mix: Dict[str, Any]
    fuel_prices: Dict[str, Any]
    policy_impact: Dict[str, Any]
    economic_indicators: Dict[str, Any]
    weather_factors: Dict[str, Any]
    risk_factors: Dict[str, Any]
    metadata: Dict[str, Any]


class FundamentalsEngine:
    """Calculate fundamental drivers for scenario execution.

    Produces load forecasts, generation mix projections, fuel forward curves,
    policy impacts, macro indicators, weather drivers, and composite risk.
    """

    def __init__(self) -> None:
        self.default_forecast_horizon = 10

    async def run(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fundamentals pipeline using provided assumptions.

        Args:
            assumptions: Structured assumptions dict (macro, fuels, power, policy, weather).

        Returns:
            Dict with fundamentals components and metadata.
        """
        try:
            load_task = asyncio.create_task(self._calculate_load_forecast(assumptions))
            supply_task = asyncio.create_task(self._calculate_generation_stack(assumptions))
            fuel_task = asyncio.create_task(self._calculate_fuel_prices(assumptions))
            policy_task = asyncio.create_task(self._assess_policy_impacts(assumptions))
            macro_task = asyncio.create_task(self._calculate_macro_indicators(assumptions))
            weather_task = asyncio.create_task(self._calculate_weather_drivers(assumptions))
            risk_task = asyncio.create_task(self._calculate_risk_factors(assumptions))

            load_forecast, generation_mix, fuel_prices, policy_impact, economic_indicators, weather_factors, risk_factors = await asyncio.gather(
                load_task,
                supply_task,
                fuel_task,
                policy_task,
                macro_task,
                weather_task,
                risk_task,
            )

            fundamentals = FundamentalsResult(
                load_forecast=load_forecast,
                generation_mix=generation_mix,
                fuel_prices=fuel_prices,
                policy_impact=policy_impact,
                economic_indicators=economic_indicators,
                weather_factors=weather_factors,
                risk_factors=risk_factors,
                metadata={
                    "methodology": "fundamentals_v2",
                    "confidence_level": 0.9,
                },
            )

            return fundamentals.__dict__
        except Exception as exc:
            logger.exception("Fundamentals layer failed, using fallback", exc_info=exc)
            return self._fallback(assumptions)

    async def _calculate_load_forecast(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Build regional load forecast and peaks from growth and sensitivities.

        Returns:
            Dict with regional projections, overall growth, and horizon.
        """
        power = assumptions.get("power", {})
        base_growth = power.get("load_growth", 1.5)
        horizon = power.get("forecast_horizon", self.default_forecast_horizon)

        regions = power.get(
            "regional_growth",
            {
                "PJM": base_growth * 1.05,
                "MISO": base_growth * 0.95,
                "CAISO": base_growth * 1.2,
                "ERCOT": base_growth * 1.15,
            },
        )

        elasticity = power.get("elasticity", -0.2)
        weather_sensitivity = power.get("weather_sensitivity", 0.3)

        projections: Dict[str, Dict[str, Any]] = {}
        for region, growth in regions.items():
            projected = 150_000 * (1 + growth / 100) ** np.arange(horizon + 1)
            projections[region] = {
                "base": 150_000,
                "projected": projected.tolist(),
                "peak": float(projected[-1] * (1 + weather_sensitivity)),
                "elasticity": elasticity,
            }

        return {
            "regions": projections,
            "overall_growth": base_growth,
            "horizon_years": horizon,
        }

    async def _calculate_generation_stack(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Project generation mix trajectory with build/retirement assumptions.

        Returns:
            Dict with current mix, projected mix by year, and reserve margin.
        """
        power = assumptions.get("power", {})
        current_mix = power.get(
            "current_mix",
            {
                "coal": 0.18,
                "gas": 0.38,
                "nuclear": 0.19,
                "hydro": 0.07,
                "wind": 0.09,
                "solar": 0.06,
                "storage": 0.03,
            },
        )

        build_rates = power.get(
            "build_rates",
            {
                "wind": 3.0,
                "solar": 5.0,
                "storage": 7.0,
                "gas": 1.0,
            },
        )

        retirements = power.get(
            "retirements",
            {
                "coal": 8.0,
                "oil": 10.0,
            },
        )

        horizon = power.get("supply_horizon", self.default_forecast_horizon)
        mix_by_year: Dict[str, Dict[str, float]] = {}

        mix = current_mix.copy()
        for year in range(horizon + 1):
            for tech, rate in build_rates.items():
                if tech in mix:
                    mix[tech] *= (1 + rate / 100)
            for tech, rate in retirements.items():
                if tech in mix:
                    mix[tech] *= max(0.0, (1 - rate / 100))

            total = sum(mix.values())
            for tech in mix:
                mix[tech] /= total

            mix_by_year[f"year_{year}"] = {tech: float(value) for tech, value in mix.items()}

        reserve_margin = power.get("reserve_margin", 0.15)

        return {
            "current_mix": current_mix,
            "projected_mix": mix_by_year,
            "reserve_margin": reserve_margin,
        }

    async def _calculate_fuel_prices(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Construct fuel forward curves from growth and volatility inputs.

        Returns:
            Dict with current prices, growth rates, volatility, and forward curves.
        """
        fuels = assumptions.get("fuels", {})
        current = fuels.get(
            "current_prices",
            {
                "natural_gas": 3.4,
                "coal": 2.1,
                "oil": 75.0,
                "uranium": 44.0,
            },
        )

        growth = fuels.get(
            "price_growth",
            {"natural_gas": 1.5, "coal": -1.0, "oil": 1.0, "uranium": 0.6},
        )

        volatility = fuels.get(
            "volatility",
            {"natural_gas": 0.25, "coal": 0.15, "oil": 0.30, "uranium": 0.12},
        )

        horizon = fuels.get("forecast_horizon", 5)

        forward_curves: Dict[str, Dict[str, float]] = {}
        for commodity, price in current.items():
            curve = {}
            for year in range(horizon + 1):
                projected = price * (1 + growth.get(commodity, 0.0) / 100) ** year
                stochastic = projected * np.exp(-0.5 * volatility[commodity] ** 2)
                curve[f"year_{year}"] = float(stochastic)
            forward_curves[commodity] = curve

        return {
            "current_prices": current,
            "growth_rates": growth,
            "volatility": volatility,
            "forward_curves": forward_curves,
        }

    async def _assess_policy_impacts(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate policy-related inputs into a normalized impact score.

        Returns:
            Dict with individual policy inputs and composite impact score.
        """
        policy = assumptions.get("policy", {})

        carbon = policy.get("carbon_price", 0.0)
        rps_targets = policy.get(
            "rps_targets",
            {"PJM": 0.25, "MISO": 0.2, "CAISO": 0.6, "ERCOT": 0.35},
        )

        transmission = policy.get("transmission_investment", 1.0)

        impact_score = (
            min(0.35, carbon / 100)
            + min(0.45, np.mean(list(rps_targets.values())) * 0.9)
            + min(0.2, max(0.0, transmission - 1.0) * 0.4)
        )

        return {
            "carbon_price": carbon,
            "rps_targets": rps_targets,
            "transmission_investment": transmission,
            "impact_score": float(min(1.0, impact_score)),
        }

    async def _calculate_macro_indicators(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract macroeconomic indicators with sane defaults."""
        macro = assumptions.get("macro", {})
        return {
            "gdp_growth": macro.get("gdp_growth", 2.4),
            "inflation": macro.get("inflation", 2.1),
            "interest_rate": macro.get("interest_rate", 3.7),
            "demand_elasticity": macro.get("demand_elasticity", -0.35),
            "industrial_growth": macro.get("industrial_growth", 2.8),
        }

    async def _calculate_weather_drivers(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble weather drivers and seasonal patterns."""
        weather = assumptions.get("weather", {})
        return {
            "temperature_sensitivity": weather.get("temperature_sensitivity", 0.65),
            "precipitation_impact": weather.get("precipitation_impact", 0.25),
            "extreme_weather_probability": weather.get("extreme_weather", 0.07),
            "seasonal_patterns": {
                "summer_peak": 1.3,
                "winter_peak": 1.18,
                "shoulder": 0.95,
            },
        }

    async def _calculate_risk_factors(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine fuel and macro volatilities into composite risk factor."""
        macro = assumptions.get("macro", {})
        fuels = assumptions.get("fuels", {})

        fuel_volatility = fuels.get("volatility", {"natural_gas": 0.25})
        macro_volatility = macro.get("volatility", 0.15)

        return {
            "fuel_risk": float(np.mean(list(fuel_volatility.values()))),
            "macro_risk": macro_volatility,
            "composite_risk": float(0.6 * macro_volatility + 0.4 * np.mean(list(fuel_volatility.values()))),
        }

    def _fallback(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "load_forecast": {"status": "fallback", "growth": assumptions.get("power", {}).get("load_growth", 1.5)},
            "generation_mix": {"status": "fallback"},
            "fuel_prices": {"status": "fallback"},
            "policy_impact": assumptions.get("policy", {}),
            "economic_indicators": assumptions.get("macro", {}),
            "weather_factors": assumptions.get("weather", {}),
            "risk_factors": {"status": "fallback"},
            "metadata": {"status": "fallback"},
        }
