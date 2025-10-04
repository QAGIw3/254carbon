"""Batch job runner for renewables analytics."""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional

from fastapi import HTTPException

from ..renewables_api import (
    BiodieselSpreadRequest,
    CarbonIntensityRequest,
    PolicyImpactRequest,
    PolicyScenario,
    RINForecastRequest,
    run_biodiesel_spread,
    run_carbon_intensity,
    run_policy_impact,
    run_rin_forecast,
)


logger = logging.getLogger(__name__)

DEFAULT_RIN_CATEGORIES = ["D4", "D5", "D6"]
DEFAULT_POLICY_SCENARIOS = [
    PolicyScenario(
        name="RFS_Tightening",
        description="Increase RIN demand by 5%",
        rin_category="all",
        demand_impact=0.05,
        supply_impact=0.0,
    ),
    PolicyScenario(
        name="BTC_Expiry",
        description="Biodiesel tax credit expires",
        incentive_changes={"federal_biodiesel_credit": -1.0},
    ),
]
DEFAULT_CARBON_INTENSITY_CONFIG = [
    {"fuel_type": "gasoline", "pathway": "conventional"},
    {"fuel_type": "biodiesel_soy", "pathway": "soy_biodiesel"},
    {"fuel_type": "ethanol_corn", "pathway": "corn_ethanol"},
]


def run_daily_renewables_jobs(
    *,
    as_of_date: Optional[date] = None,
    model_version: str = "v1",
) -> Dict[str, int]:
    """Run renewables analytics batch jobs."""

    as_of = as_of_date or date.today()
    totals = {
        "rin_forecast": 0,
        "biodiesel_spread": 0,
        "carbon_intensity": 0,
        "policy_impact": 0,
    }

    logger.info("Running RIN price forecast batch for %s", as_of)
    try:
        rin_response = run_rin_forecast(
            RINForecastRequest(as_of_date=as_of, categories=DEFAULT_RIN_CATEGORIES, horizon_days=90, model_version=model_version)
        )
        totals["rin_forecast"] += rin_response.persisted_rows
    except HTTPException as exc:
        logger.warning("RIN forecast skipped: %s", exc.detail)
    except Exception:
        logger.exception("RIN forecast batch failure")

    logger.info("Running biodiesel spread batch for %s", as_of)
    try:
        biodiesel_response = run_biodiesel_spread(
            BiodieselSpreadRequest(as_of_date=as_of, region=None, model_version=model_version)
        )
        totals["biodiesel_spread"] += biodiesel_response.persisted_rows
    except HTTPException as exc:
        logger.warning("Biodiesel spread skipped: %s", exc.detail)
    except Exception:
        logger.exception("Biodiesel spread batch failure")

    for config in DEFAULT_CARBON_INTENSITY_CONFIG:
        logger.info("Running carbon intensity batch for %s/%s", config["fuel_type"], config["pathway"])
        try:
            ci_response = run_carbon_intensity(
                CarbonIntensityRequest(
                    as_of_date=as_of,
                    fuel_type=config["fuel_type"],
                    pathway=config["pathway"],
                    model_version=model_version,
                )
            )
            totals["carbon_intensity"] += ci_response.persisted_rows
        except HTTPException as exc:
            logger.warning(
                "Carbon intensity skipped for %s: %s",
                config["fuel_type"],
                exc.detail,
            )
        except Exception:
            logger.exception(
                "Carbon intensity batch failure for %s",
                config["fuel_type"],
            )

    logger.info("Running renewables policy impact batch")
    try:
        policy_response = run_policy_impact(
            PolicyImpactRequest(
                as_of_date=as_of,
                scenarios=DEFAULT_POLICY_SCENARIOS,
                rin_categories=DEFAULT_RIN_CATEGORIES,
                model_version=model_version,
            )
        )
        totals["policy_impact"] += policy_response.persisted_rows
    except HTTPException as exc:
        logger.warning("Policy impact skipped: %s", exc.detail)
    except Exception:
        logger.exception("Policy impact batch failure")

    logger.info("Renewables batch complete with totals: %s", totals)
    return totals


__all__ = ["run_daily_renewables_jobs"]

