"""
WebSocket authentication helpers for validating JWTs outside FastAPI dependency injection.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import jwt

# Import using absolute path since we're in the gateway package
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import auth

logger = logging.getLogger(__name__)


async def verify_ws_token(token: str) -> Dict[str, Any]:
    """Validate a raw JWT string for WebSocket connections.

    Performs JWKS lookup and claim verification matching REST auth; useful
    when FastAPI dependency injection is not available on WS upgrades.

    Args:
        token: Raw bearer token string.

    Returns:
        Dict of normalized user claims on success.

    Raises:
        ValueError: If the token is missing required headers/claims or invalid.
    """
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid:
            raise ValueError("Token missing key ID")

        jwks = await auth.get_jwks()
        key = None
        for k in jwks.get("keys", []):
            if k.get("kid") == kid:
                key = k
                break
        if not key:
            raise ValueError("Unknown key ID")

        if key.get("kty") != "RSA":
            raise ValueError("Unsupported key type")

        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=auth.KEYCLOAK_AUDIENCE,
            issuer=f"{auth.KEYCLOAK_URL}/",
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
            },
        )

        exp = payload.get("exp")
        if exp and exp < time.time():
            raise ValueError("Token expired")

        return {
            "sub": payload.get("sub"),
            "email": payload.get("email"),
            "name": payload.get("name"),
            "tenant_id": payload.get("tenant_id"),
            "roles": payload.get("realm_access", {}).get("roles", []),
            "scopes": payload.get("scope", "").split(),
            "groups": payload.get("groups", []),
        }

    except Exception as e:
        logger.warning(f"WebSocket token validation failed: {e}")
        raise

