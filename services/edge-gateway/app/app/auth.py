"""Authentication and authorization using Keycloak OIDC."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

security = HTTPBearer()

KEYCLOAK_URL = "http://keycloak:8080/auth/realms/254carbon"
KEYCLOAK_AUDIENCE = "market-intelligence-api"

_jwks_cache: Optional[Dict[str, Any]] = None
_jwks_cache_time: Optional[datetime] = None
_CACHE_DURATION = timedelta(hours=1)


async def get_jwks() -> Dict[str, Any]:
    global _jwks_cache, _jwks_cache_time

    now = datetime.utcnow()

    if _jwks_cache and _jwks_cache_time and (now - _jwks_cache_time) < _CACHE_DURATION:
        return _jwks_cache

    jwks_url = f"{KEYCLOAK_URL}/protocol/openid-connect/certs"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(jwks_url)
            response.raise_for_status()
    except Exception as exc:
        logger.error("Failed to fetch JWKS: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication service unavailable") from exc

    _jwks_cache = response.json()
    _jwks_cache_time = now
    return _jwks_cache


def find_key_by_kid(kid: str, jwks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    return None


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    token = credentials.credentials

    try:
        header = jwt.get_unverified_header(token)
        if not header.get("kid"):
            raise HTTPException(status_code=401, detail="Token missing key ID")

        jwks = await get_jwks()
        key = find_key_by_kid(header["kid"], jwks)
        if not key:
            raise HTTPException(status_code=401, detail="Invalid key ID")

        if key.get("kty") != "RSA":
            raise HTTPException(status_code=401, detail="Unsupported key type")

        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=KEYCLOAK_AUDIENCE,
            issuer=f"{KEYCLOAK_URL}/",
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
            },
        )

        exp = payload.get("exp")
        if exp and exp < time.time():
            raise HTTPException(status_code=401, detail="Token expired")

        return {
            "sub": payload.get("sub"),
            "email": payload.get("email"),
            "name": payload.get("name"),
            "tenant_id": payload.get("tenant_id"),
            "roles": payload.get("realm_access", {}).get("roles", []),
            "scopes": payload.get("scope", "").split(),
            "groups": payload.get("groups", []),
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid token issuer")
    except jwt.InvalidSignatureError:
        raise HTTPException(status_code=401, detail="Invalid token signature")
    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token format")
    except Exception as exc:
        logger.error("Token verification error: %s", exc)
        raise HTTPException(status_code=401, detail="Authentication failed") from exc


def has_permission(user: Dict[str, Any], permission: str) -> bool:
    return permission in user.get("scopes", [])


def has_role(user: Dict[str, Any], role: str) -> bool:
    return role in user.get("roles", [])
