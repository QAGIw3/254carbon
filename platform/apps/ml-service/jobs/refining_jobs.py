"""Batch job runner for refining analytics."""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Iterable, List, Optional

from fastapi import HTTPException

from ..refining_api import (
    CrackOptimizeRequest,
    ElasticityRequest,
    FuelSubstitutionRequest,
    RefineryYieldRequest,
    run_crack_optimization,
    run_product_elasticity,
    run_refinery_yield,
    run_fuel_substitution,
)


logger = logging.getLogger(__name__)

DEFAULT_REFINING_REGIONS = ["PADD1", "PADD3"]
DEFAULT_CRACK_TYPES = ["3:2:1", "5:3:2"]
DEFAULT_CRUDE_BY_REGION = {
    "PADD1": "OIL.BRENT",
    "PADD3": "OIL.WTI",
}
DEFAULT_ELASTICITY_CONFIG = [
    {
        "product": "gasoline",
        "price_instrument_id": "OIL.RBOB",
        "demand_variable": "gasoline_demand",
    },
    {
        "product": "diesel",
        "price_instrument_id": "OIL.ULSD",
        "demand_variable": "diesel_demand",
    },
]
DEFAULT_SUBSTITUTION_REGIONS = ["US"]


def _resolve_crude(region: str) -> str:
    return DEFAULT_CRUDE_BY_REGION.get(region, "OIL.WTI")


def run_daily_refining_jobs(
    *,
    as_of_date: Optional[date] = None,
    regions: Optional[Iterable[str]] = None,
    model_version: str = "v1",
) -> Dict[str, int]:
    """Run refining analytics batch jobs for the supplied regions."""

    as_of = as_of_date or date.today()
    region_list = list(regions) if regions else DEFAULT_REFINING_REGIONS

    totals = {
        "crack": 0,
        "yields": 0,
        "elasticities": 0,
        "substitution": 0,
    }

    for region in region_list:
        logger.info("Running crack optimization for %s (%s)", region, as_of)
        try:
            crack_response = run_crack_optimization(
                CrackOptimizeRequest(
                    as_of_date=as_of,
                    region=region,
                    crack_types=DEFAULT_CRACK_TYPES,
                    crude_code=_resolve_crude(region),
                    model_version=model_version,
                )
            )
            totals["crack"] += crack_response.persisted_rows
        except HTTPException as exc:
            logger.warning("Crack optimization skipped for %s: %s", region, exc.detail)
        except Exception:
            logger.exception("Crack optimization failure for region %s", region)

        logger.info("Running refinery yields for %s", region)
        try:
            yield_response = run_refinery_yield(
                RefineryYieldRequest(
                    as_of_date=as_of,
                    crude_type=_resolve_crude(region).split(".")[-1].lower(),
                    region=region,
                    model_version=model_version,
                )
            )
            totals["yields"] += yield_response.persisted_rows
        except HTTPException as exc:
            logger.warning("Refinery yield skipped for %s: %s", region, exc.detail)
        except Exception:
            logger.exception("Refinery yield failure for region %s", region)

    # Elasticities run at national level to ensure data availability
    for elasticity_cfg in DEFAULT_ELASTICITY_CONFIG:
        logger.info("Running demand elasticity for %s", elasticity_cfg["product"])
        try:
            elasticity_response = run_product_elasticity(
                ElasticityRequest(
                    as_of_date=as_of,
                    product=elasticity_cfg["product"],
                    region="US",
                    method="regression",
                    price_instrument_id=elasticity_cfg["price_instrument_id"],
                    demand_entity_id="US",
                    demand_variable=elasticity_cfg["demand_variable"],
                    model_version=model_version,
                )
            )
            totals["elasticities"] += elasticity_response.persisted_rows
        except HTTPException as exc:
            logger.warning(
                "Demand elasticity skipped for %s: %s",
                elasticity_cfg["product"],
                exc.detail,
            )
        except Exception:
            logger.exception(
                "Demand elasticity failure for product %s",
                elasticity_cfg["product"],
            )

    for region in DEFAULT_SUBSTITUTION_REGIONS:
        logger.info("Running fuel substitution metrics for %s", region)
        try:
            substitution_response = run_fuel_substitution(
                FuelSubstitutionRequest(
                    as_of_date=as_of,
                    region=region,
                    demand_entity_id="US",
                    model_version=model_version,
                )
            )
            totals["substitution"] += substitution_response.persisted_rows
        except HTTPException as exc:
            logger.warning("Fuel substitution skipped for %s: %s", region, exc.detail)
        except Exception:
            logger.exception("Fuel substitution failure for region %s", region)

    logger.info("Refining batch complete with totals: %s", totals)
    return totals


__all__ = ["run_daily_refining_jobs"]

