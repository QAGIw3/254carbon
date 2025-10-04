"""Batch job runners for energy transition and carbon analytics."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

from ..carbon_api import (
    CarbonLeakageRiskRequest,
    CarbonPriceForecastRequest,
    ComplianceCostRequest,
    EmissionObservation,
    PolicyImpactRequest,
    run_carbon_leakage_risk,
    run_carbon_price_forecast,
    run_compliance_costs,
    run_policy_impact,
)
from ..transition_api import (
    DecarbonizationPathwayRequest,
    PolicyScenarioInput,
    RenewableAdoptionRequest,
    StrandedAssetRiskRequest,
    run_decarbonization_pathway,
    run_renewable_adoption,
    run_stranded_asset_risk,
)


logger = logging.getLogger(__name__)

DEFAULT_CARBON_MARKETS = ["eua", "cca", "rggi"]
DEFAULT_POLICY_SCENARIOS = [
    PolicyScenarioInput(
        name="HighPolicyTightening",
        description="Higher policy tightening across markets",
        price_impact=0.08,
        market_factors={"eua": 1.1, "cca": 1.05, "rggi": 1.03},
    ),
    PolicyScenarioInput(
        name="RecessionSoftening",
        description="Demand shock reduces allowance prices",
        price_impact=-0.06,
        market_factors={"eua": 0.92, "cca": 0.9, "rggi": 0.88},
    ),
]
DEFAULT_TRANSITION_SCENARIOS = ["conservative", "moderate", "ambitious"]
DEFAULT_RENEWABLE_TECHS = [
    ("solar", 120.0, 1.3),
    ("wind", 95.0, 1.2),
    ("battery_storage", 40.0, 1.5),
]


def run_nightly_carbon_jobs(
    *,
    as_of_date: Optional[date] = None,
    model_version: str = "v1",
) -> Dict[str, int]:
    """Nightly carbon analytics batch job."""

    as_of = as_of_date or date.today()
    totals = {"price_forecast": 0, "policy_impact": 0}

    logger.info("Running nightly carbon price forecast batch for %s", as_of)
    forecast_response = run_carbon_price_forecast(
        CarbonPriceForecastRequest(
            as_of_date=as_of,
            markets=DEFAULT_CARBON_MARKETS,
            policy_scenarios=DEFAULT_POLICY_SCENARIOS,
            model_version=model_version,
        )
    )
    totals["price_forecast"] += forecast_response.persisted_rows

    logger.info("Running nightly carbon policy impact batch for %s", as_of)
    policy_response = run_policy_impact(
        PolicyImpactRequest(
            as_of_date=as_of,
            baseline_prices={
                "eua": 85.0,
                "cca": 32.0,
                "rggi": 14.0,
            },
            policy_scenarios=DEFAULT_POLICY_SCENARIOS,
            model_version=model_version,
        )
    )
    totals["policy_impact"] += policy_response.persisted_rows

    logger.info("Nightly carbon batch complete: %s", totals)
    return totals


def run_weekly_transition_jobs(
    *,
    as_of_date: Optional[date] = None,
    model_version: str = "v1",
) -> Dict[str, int]:
    """Weekly transition batch covering pathways and renewable adoption."""

    as_of = as_of_date or date.today()
    totals = {"decarbonization": 0, "renewable_adoption": 0}

    for sector in ("power", "transportation", "industry", "buildings"):
        for scenario in DEFAULT_TRANSITION_SCENARIOS:
            logger.info(
                "Running decarbonization pathway batch for %s/%s", sector, scenario
            )
            response = run_decarbonization_pathway(
                DecarbonizationPathwayRequest(
                    as_of_date=as_of,
                    sector=sector,
                    policy_scenario=scenario,
                    model_version=model_version,
                )
            )
            totals["decarbonization"] += response.persisted_rows

    for technology, capacity, policy_support in DEFAULT_RENEWABLE_TECHS:
        logger.info(
            "Running renewable adoption batch for %s", technology
        )
        adoption_response = run_renewable_adoption(
            RenewableAdoptionRequest(
                as_of_date=as_of,
                technology=technology,
                current_capacity=capacity,
                policy_support=policy_support,
                model_version=model_version,
            )
        )
        totals["renewable_adoption"] += adoption_response.persisted_rows

    logger.info("Weekly transition batch complete: %s", totals)
    return totals


def run_monthly_stranded_asset_job(
    *,
    as_of_date: Optional[date] = None,
    model_version: str = "v1",
) -> Dict[str, int]:
    """Monthly stranded asset analytics batch job."""

    as_of = as_of_date or date.today()
    totals = {"stranded_asset_risk": 0}

    asset_values = {
        "coal_generation": 2_500_000_000.0,
        "gas_generation": 1_800_000_000.0,
        "refining_assets": 1_200_000_000.0,
    }
    lifetimes = {
        "coal_generation": 15,
        "gas_generation": 18,
        "refining_assets": 12,
    }

    response = run_stranded_asset_risk(
        StrandedAssetRiskRequest(
            as_of_date=as_of,
            asset_values=asset_values,
            asset_lifetimes=lifetimes,
            policy_scenarios=DEFAULT_POLICY_SCENARIOS,
            model_version=model_version,
        )
    )
    totals["stranded_asset_risk"] += response.persisted_rows

    logger.info("Monthly stranded asset batch complete: %s", totals)
    return totals


def run_weekly_carbon_compliance_jobs(
    *,
    as_of_date: Optional[date] = None,
    model_version: str = "v1",
) -> Dict[str, int]:
    """Weekly carbon compliance and leakage analytics batch job."""

    as_of = as_of_date or date.today()
    totals = {"compliance_costs": 0, "leakage_risk": 0}

    def _emissions_curve(base: float) -> List[EmissionObservation]:
        return [
            EmissionObservation(period=as_of - timedelta(weeks=4 * idx), emissions=base * (0.95 ** idx))
            for idx in range(6)
        ]

    compliance_response = run_compliance_costs(
        ComplianceCostRequest(
            as_of_date=as_of,
            markets=DEFAULT_CARBON_MARKETS,
            emissions_data={
                "eua": _emissions_curve(60_000_000),
                "cca": _emissions_curve(25_000_000),
                "rggi": _emissions_curve(12_000_000),
            },
            compliance_obligations={
                "power": 0.45,
                "industry": 0.35,
                "transport": 0.20,
            },
            policy_scenarios=DEFAULT_POLICY_SCENARIOS,
            model_version=model_version,
        )
    )
    totals["compliance_costs"] += compliance_response.persisted_rows

    leakage_response = run_carbon_leakage_risk(
        CarbonLeakageRiskRequest(
            as_of_date=as_of,
            domestic_prices={"steel": 95.0, "cement": 82.0, "chemicals": 74.0},
            international_prices={"steel": 45.0, "cement": 40.0, "chemicals": 38.0},
            trade_exposure={"steel": 0.65, "cement": 0.4, "chemicals": 0.55},
            emissions_intensity={"steel": 2.3, "cement": 0.9, "chemicals": 1.4},
            model_version=model_version,
        )
    )
    totals["leakage_risk"] += leakage_response.persisted_rows

    logger.info("Weekly carbon compliance batch complete: %s", totals)
    return totals


__all__ = [
    "run_nightly_carbon_jobs",
    "run_weekly_transition_jobs",
    "run_monthly_stranded_asset_job",
    "run_weekly_carbon_compliance_jobs",
]

