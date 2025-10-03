"""
WebSocket authentication helpers for validating JWTs outside FastAPI dependency injection.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict

import jwt

# Add current directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Use absolute import
from auth import get_jwks, KEYCLOAK_AUDIENCE, KEYCLOAK_URL

logger = logging.getLogger(__name__)


async def verify_ws_token(token: str) -> Dict[str, Any]:
    """Validate a raw JWT string for WebSocket connections and return user claims.

    Raises ValueError on invalid tokens.
    """
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid:
            raise ValueError("Token missing key ID")

        jwks = await get_jwks()
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


