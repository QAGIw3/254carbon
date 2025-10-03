"""
Authentication and authorization using Keycloak OIDC.
"""
import logging
from typing import Optional

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Keycloak configuration (loaded from env)
KEYCLOAK_URL = "http://keycloak:8080/auth/realms/254carbon"
KEYCLOAK_AUDIENCE = "market-intelligence-api"


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """
    Verify JWT token from Keycloak.
    
    Returns decoded token payload with user information.
    """
    token = credentials.credentials
    
    try:
        # TODO: Implement full OIDC token verification with Keycloak
        # For now, decode without verification (development only)
        payload = jwt.decode(
            token,
            options={"verify_signature": False},  # SECURITY: Enable in production
        )
        
        return {
            "sub": payload.get("sub"),
            "email": payload.get("email"),
            "tenant_id": payload.get("tenant_id"),
            "roles": payload.get("realm_access", {}).get("roles", []),
            "scopes": payload.get("scope", "").split(),
        }
    except jwt.DecodeError as e:
        logger.error(f"Token decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


def has_permission(user: dict, permission: str) -> bool:
    """
    Check if user has required permission.
    
    Permissions format: "read:ticks", "write:scenarios", etc.
    """
    scopes = user.get("scopes", [])
    return permission in scopes


def has_role(user: dict, role: str) -> bool:
    """Check if user has required role."""
    roles = user.get("roles", [])
    return role in roles

