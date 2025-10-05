"""Versioned API composition for analytics platform."""

from fastapi import APIRouter

from . import legacy


router = APIRouter()
router.include_router(legacy.router)

