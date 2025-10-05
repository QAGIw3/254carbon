"""Adapter router that temporarily wraps legacy analytics service routers."""

from fastapi import APIRouter

from ..legacy.routers import (
    arbitrage_api,
    carbon_api,
    insights,
    portfolio_api,
    quantum,
    refining_api,
    renewables_api,
    research_api,
    satellite,
    supply_chain_api,
    transition_api,
)


router = APIRouter()
router.include_router(arbitrage_api.router)
router.include_router(carbon_api.router)
router.include_router(insights.router)
router.include_router(portfolio_api.router)
router.include_router(quantum.router)
router.include_router(refining_api.router)
router.include_router(renewables_api.router)
router.include_router(research_api.router)
router.include_router(satellite.router)
router.include_router(supply_chain_api.router)
router.include_router(transition_api.router)

