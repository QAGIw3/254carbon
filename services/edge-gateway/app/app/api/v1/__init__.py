"""API v1 router."""

from fastapi import APIRouter

from . import routes

router = APIRouter(prefix="", tags=["health"])
router.include_router(routes.router)

