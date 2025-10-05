"""
Auth Service - Centralized authentication for 254Carbon platform

Provides JWT validation and user claims extraction for all microservices.
Integrates with Keycloak OIDC for token verification.

Port: 8010
"""
import logging
import os
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from jwt_handler import verify_jwt_token, verify_ws_jwt_token
from keycloak import get_jwks, clear_jwks_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="254Carbon Auth Service",
    description="Centralized authentication service with Keycloak integration",
    version="1.0.0",
)


# Models
class TokenVerifyRequest(BaseModel):
    token: str


class TokenVerifyResponse(BaseModel):
    valid: bool
    user_claims: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    keycloak_reachable: bool


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Test Keycloak connectivity
    keycloak_reachable = True
    try:
        await get_jwks()
    except Exception:
        keycloak_reachable = False
    
    return HealthResponse(
        status="healthy" if keycloak_reachable else "degraded",
        timestamp=datetime.utcnow(),
        keycloak_reachable=keycloak_reachable,
    )


@app.post("/auth/verify", response_model=TokenVerifyResponse)
async def verify_token(request: TokenVerifyRequest):
    """
    Verify a JWT token (REST API usage).
    
    Validates token signature, expiration, audience, and issuer.
    Returns user claims on success.
    """
    try:
        user_claims = await verify_jwt_token(request.token)
        
        return TokenVerifyResponse(
            valid=True,
            user_claims=user_claims,
        )
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        return TokenVerifyResponse(
            valid=False,
            error=str(e),
        )


@app.post("/auth/verify-ws", response_model=TokenVerifyResponse)
async def verify_websocket_token(request: TokenVerifyRequest):
    """
    Verify a JWT token for WebSocket connections.
    
    Similar to /auth/verify but optimized for WebSocket auth flow.
    """
    try:
        user_claims = await verify_ws_jwt_token(request.token)
        
        return TokenVerifyResponse(
            valid=True,
            user_claims=user_claims,
        )
    except Exception as e:
        logger.warning(f"WebSocket token verification failed: {e}")
        return TokenVerifyResponse(
            valid=False,
            error=str(e),
        )


@app.get("/auth/user-info")
async def get_user_info(token: str):
    """
    Get user information from token without full verification.
    
    Returns decoded claims (still validates signature and exp).
    Useful for quick user info lookup.
    """
    try:
        user_claims = await verify_jwt_token(token)
        return {
            "user_id": user_claims.get("sub"),
            "email": user_claims.get("email"),
            "name": user_claims.get("name"),
            "tenant_id": user_claims.get("tenant_id"),
            "roles": user_claims.get("roles", []),
        }
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/auth/refresh-keys")
async def refresh_jwks():
    """
    Force refresh of JWKS cache.
    
    Useful when Keycloak keys are rotated.
    """
    try:
        clear_jwks_cache()
        jwks = await get_jwks()
        
        return {
            "status": "success",
            "keys_count": len(jwks.get("keys", [])),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to refresh JWKS: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh keys")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8010"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )

