"""
Authentication and authorization using Keycloak OIDC.
"""
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import jwt
import httpx
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Keycloak configuration (loaded from env)
KEYCLOAK_URL = "http://keycloak:8080/auth/realms/254carbon"
KEYCLOAK_AUDIENCE = "market-intelligence-api"

# Cache for JWKS (JSON Web Key Set)
_jwks_cache: Optional[Dict[str, Any]] = None
_jwks_cache_time: Optional[datetime] = None
_CACHE_DURATION = timedelta(hours=1)


async def get_jwks() -> Dict[str, Any]:
    """Fetch JWKS from Keycloak with caching.

    Returns:
        Dict[str, Any]: JWKS payload as returned by the authorization server.
    """
    global _jwks_cache, _jwks_cache_time

    now = datetime.utcnow()

    # Return cached JWKS if still valid
    if _jwks_cache and _jwks_cache_time and (now - _jwks_cache_time) < _CACHE_DURATION:
        return _jwks_cache

    try:
        # Fetch JWKS from Keycloak
        jwks_url = f"{KEYCLOAK_URL}/protocol/openid-connect/certs"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(jwks_url)
            response.raise_for_status()

        jwks_data = response.json()

        # Cache the JWKS
        _jwks_cache = jwks_data
        _jwks_cache_time = now

        logger.info("JWKS fetched and cached successfully")
        return jwks_data

    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise HTTPException(status_code=500, detail="Authentication service unavailable")


def find_key_by_kid(kid: str, jwks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find the key with matching kid in JWKS.

    Args:
        kid: Key ID from the token header.
        jwks: JWKS document containing keys.

    Returns:
        Matching key dict if found, otherwise None.
    """
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    return None


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """Verify a JWT token issued by Keycloak.

    Performs header inspection, JWKS lookup, RSA signature verification and
    claim validation (exp, iat, aud, iss). Returns selected user claims.

    Args:
        credentials: FastAPI security-provided bearer credentials.

    Returns:
        dict: Normalized user claims for downstream authorization.
    """
    token = credentials.credentials

    try:
        # Decode token header to get kid (key ID)
        unverified_header = jwt.get_unverified_header(token)

        if not unverified_header.get("kid"):
            raise HTTPException(status_code=401, detail="Token missing key ID")

        # Get JWKS for signature verification
        jwks = await get_jwks()

        # Find the correct key
        key = find_key_by_kid(unverified_header["kid"], jwks)
        if not key:
            raise HTTPException(status_code=401, detail="Invalid key ID")

        # Extract public key components
        if key.get("kty") != "RSA":
            raise HTTPException(status_code=401, detail="Unsupported key type")

        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)

        # Verify and decode token
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
            }
        )

        # Check token expiration
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
        logger.warning("Token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidAudienceError:
        logger.warning("Invalid audience in token")
        raise HTTPException(status_code=401, detail="Invalid token audience")
    except jwt.InvalidIssuerError:
        logger.warning("Invalid issuer in token")
        raise HTTPException(status_code=401, detail="Invalid token issuer")
    except jwt.InvalidSignatureError:
        logger.warning("Invalid token signature")
        raise HTTPException(status_code=401, detail="Invalid token signature")
    except jwt.DecodeError as e:
        logger.error(f"Token decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token format")
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


def has_permission(user: dict, permission: str) -> bool:
    """Check if user has a required permission.

    Permissions format: "read:ticks", "write:scenarios", etc.

    Args:
        user: Verified user claims mapping.
        permission: Permission string to check.

    Returns:
        True if the permission is present in user scopes.
    """
    scopes = user.get("scopes", [])
    return permission in scopes


def has_role(user: dict, role: str) -> bool:
    """Check if user has required role.

    Args:
        user: Verified user claims mapping.
        role: Role name to check.

    Returns:
        True if the role is present in user roles.
    """
    roles = user.get("roles", [])
    return role in roles
