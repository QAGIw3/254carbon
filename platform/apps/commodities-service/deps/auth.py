"""
Minimal auth helpers mirroring the gateway service.

This module provides a minimal JWT verification path against Keycloak where
available, with a `LOCAL_DEV` bypass that injects permissive claims. Routers
use `require_roles()` to perform simple role checks on incoming requests.

The gateway continues to be the primary AuthN/Z boundary; these checks serve
as defense-in-depth and allow this service to be accessed directly where
needed (e.g., internal proxies or batch jobs).
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable

import httpx
import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

security_scheme = HTTPBearer(auto_error=False)

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080/auth/realms/254carbon")
KEYCLOAK_AUDIENCE = os.getenv("KEYCLOAK_AUDIENCE", "market-intelligence-api")
LOCAL_DEV_BYPASS = os.getenv("LOCAL_DEV", "false").lower() in {"1", "true", "yes"}

_JWKS_CACHE: Optional[Dict[str, Any]] = None
_JWKS_CACHE_EXPIRY: Optional[datetime] = None
_JWKS_TTL = timedelta(hours=1)


async def _fetch_jwks() -> Dict[str, Any]:
    """Fetch JWKS from Keycloak with simple in-process caching."""

    global _JWKS_CACHE, _JWKS_CACHE_EXPIRY

    if _JWKS_CACHE and _JWKS_CACHE_EXPIRY and datetime.utcnow() < _JWKS_CACHE_EXPIRY:
        return _JWKS_CACHE

    jwks_url = f"{KEYCLOAK_URL}/protocol/openid-connect/certs"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(jwks_url)
        response.raise_for_status()
        payload = response.json()

    _JWKS_CACHE = payload
    _JWKS_CACHE_EXPIRY = datetime.utcnow() + _JWKS_TTL
    return payload


def _find_key(kid: str, jwks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    return None


async def _verify_token(token: str) -> Dict[str, Any]:
    """Verify RS256 JWT using JWKS and return normalized claims."""
    try:
        unverified_header = jwt.get_unverified_header(token)
    except jwt.DecodeError as exc:
        raise HTTPException(status_code=401, detail="Invalid token header") from exc

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="Token missing key id")

    jwks = await _fetch_jwks()
    key = _find_key(kid, jwks)
    if not key:
        raise HTTPException(status_code=401, detail="Unknown signing key")

    if key.get("kty") != "RSA":
        raise HTTPException(status_code=401, detail="Unsupported key type")

    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=KEYCLOAK_AUDIENCE,
            issuer=f"{KEYCLOAK_URL}/",
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    return {
        "sub": payload.get("sub"),
        "email": payload.get("email"),
        "name": payload.get("name"),
        "tenant_id": payload.get("tenant_id"),
        "roles": payload.get("realm_access", {}).get("roles", []),
        "scopes": payload.get("scope", "").split(),
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
) -> Dict[str, Any]:
    """
    FastAPI dependency for authenticated user.

    Behavior
    - LOCAL_DEV: returns a permissive claim set with common roles.
    - Otherwise: requires Authorization: Bearer and validates signature,
      audience, and issuer. Only a subset of claims are returned.
    """

    if LOCAL_DEV_BYPASS:
        return {
            "sub": "local-dev",
            "tenant_id": "LOCAL",
            "roles": [
                "commodities.read",
                "gas_data_access",
                "oil_data_access",
                "coal_data_access",
                "biofuels_data_access",
                "battery_materials_access",
            ],
            "scopes": ["commodities:read"],
        }

    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    return await _verify_token(token)


def require_roles(*roles: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a dependency enforcing required roles.

    Example
    -------
    @router.get("/gas/prices")
    async def handler(user: dict = Depends(require_roles("gas_data_access"))):
        ...
    """

    async def _dependency(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_roles = set(user.get("roles", []))
        missing = [role for role in roles if role not in user_roles]
        if missing:
            raise HTTPException(status_code=403, detail=f"Missing roles: {', '.join(missing)}")
        return user

    return _dependency


__all__ = ["get_current_user", "require_roles"]
