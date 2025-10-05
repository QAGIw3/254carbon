"""API v1 router."""

from fastapi import APIRouter

from . import analytics, commodities_proxy, exports, research, routes

router = APIRouter()
router.include_router(routes.router, tags=["core"])
router.include_router(commodities_proxy.router, tags=["commodities"])
router.include_router(exports.router, tags=["exports"])
router.include_router(analytics.router, tags=["analytics"])
router.include_router(research.router, tags=["research"])

