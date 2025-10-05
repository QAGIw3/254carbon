"""
Entitlements Service - Access control for 254Carbon platform

Manages user entitlements for markets, products, and channels.
Checks tenant permissions for instruments and data access.

Port: 8011
"""
import logging
import os
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from entitlement_rules import (
    check_user_entitlement,
    get_user_entitlements_list,
    bulk_check_entitlements,
)
from db import init_db_pool, close_db_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="254Carbon Entitlements Service",
    description="Centralized entitlement and permission management",
    version="1.0.0",
)


# Models
class EntitlementCheckRequest(BaseModel):
    user_id: str
    tenant_id: str
    instrument_id: str
    channel: str  # "hub", "api", "downloads", "stream"


class EntitlementCheckResponse(BaseModel):
    entitled: bool
    reason: Optional[str] = None


class BulkEntitlementCheckRequest(BaseModel):
    user_id: str
    tenant_id: str
    checks: List[dict]  # [{instrument_id, channel}, ...]


class BulkEntitlementCheckResponse(BaseModel):
    results: List[dict]  # [{instrument_id, channel, entitled}, ...]


class UserEntitlementsResponse(BaseModel):
    user_id: str
    tenant_id: str
    entitlements: List[dict]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database_connected: bool


# Lifecycle
@app.on_event("startup")
async def startup():
    """Initialize database connection pool."""
    await init_db_pool()
    logger.info("Entitlements Service started")


@app.on_event("shutdown")
async def shutdown():
    """Close database connections."""
    await close_db_pool()
    logger.info("Entitlements Service shut down")


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from db import get_pool
    
    db_connected = True
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_connected = False
    
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp=datetime.utcnow(),
        database_connected=db_connected,
    )


@app.post("/entitlements/check", response_model=EntitlementCheckResponse)
async def check_entitlement(request: EntitlementCheckRequest):
    """
    Check if user has entitlement for instrument and channel.
    
    Validates tenant permissions for market/product/channel access.
    """
    try:
        user = {
            "sub": request.user_id,
            "tenant_id": request.tenant_id,
        }
        
        entitled = await check_user_entitlement(
            user,
            request.instrument_id,
            request.channel,
        )
        
        return EntitlementCheckResponse(
            entitled=entitled,
            reason=None if entitled else "No valid entitlement found",
        )
        
    except Exception as e:
        logger.error(f"Error checking entitlement: {e}")
        return EntitlementCheckResponse(
            entitled=False,
            reason=f"Error: {str(e)}",
        )


@app.post("/entitlements/bulk-check", response_model=BulkEntitlementCheckResponse)
async def bulk_check(request: BulkEntitlementCheckRequest):
    """
    Check multiple entitlements in a single request.
    
    Optimized for checking many instruments at once.
    """
    try:
        user = {
            "sub": request.user_id,
            "tenant_id": request.tenant_id,
        }
        
        results = await bulk_check_entitlements(user, request.checks)
        
        return BulkEntitlementCheckResponse(results=results)
        
    except Exception as e:
        logger.error(f"Error in bulk check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entitlements/user/{user_id}", response_model=UserEntitlementsResponse)
async def get_user_entitlements(user_id: str, tenant_id: str):
    """
    Get all entitlements for a user.
    
    Returns list of markets, products, and channels the user has access to.
    """
    try:
        user = {
            "sub": user_id,
            "tenant_id": tenant_id,
        }
        
        entitlements = await get_user_entitlements_list(user)
        
        return UserEntitlementsResponse(
            user_id=user_id,
            tenant_id=tenant_id,
            entitlements=entitlements,
        )
        
    except Exception as e:
        logger.error(f"Error fetching user entitlements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8011"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )

