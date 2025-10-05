"""
JWT token verification and validation.
"""
import logging
import time
from typing import Dict, Any

import jwt

from keycloak import get_jwks, find_key_by_kid, KEYCLOAK_URL, KEYCLOAK_AUDIENCE

logger = logging.getLogger(__name__)


async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify a JWT token issued by Keycloak.
    
    Performs:
    - Header inspection for kid (key ID)
    - JWKS lookup and RSA signature verification
    - Claim validation (exp, iat, aud, iss)
    
    Args:
        token: JWT token string (without "Bearer " prefix)
    
    Returns:
        dict: Normalized user claims for downstream authorization.
        
    Raises:
        jwt.ExpiredSignatureError: Token expired
        jwt.InvalidAudienceError: Wrong audience
        jwt.InvalidIssuerError: Wrong issuer
        jwt.InvalidSignatureError: Invalid signature
        jwt.DecodeError: Token decode failed
        ValueError: Token validation failed
    """
    try:
        # Decode token header to get kid (key ID)
        unverified_header = jwt.get_unverified_header(token)

        if not unverified_header.get("kid"):
            raise ValueError("Token missing key ID")

        # Get JWKS for signature verification
        jwks = await get_jwks()

        # Find the correct key
        key = find_key_by_kid(unverified_header["kid"], jwks)
        if not key:
            raise ValueError("Invalid key ID")

        # Extract public key components
        if key.get("kty") != "RSA":
            raise ValueError("Unsupported key type")

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

        # Check token expiration (double-check)
        exp = payload.get("exp")
        if exp and exp < time.time():
            raise jwt.ExpiredSignatureError("Token expired")

        # Return normalized claims
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
        raise
    except jwt.InvalidAudienceError:
        logger.warning("Invalid audience in token")
        raise
    except jwt.InvalidIssuerError:
        logger.warning("Invalid issuer in token")
        raise
    except jwt.InvalidSignatureError:
        logger.warning("Invalid token signature")
        raise
    except jwt.DecodeError as e:
        logger.error(f"Token decode error: {e}")
        raise
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise


async def verify_ws_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token for WebSocket connections.
    
    Identical to verify_jwt_token but optimized for WebSocket auth flow.
    Separate function for clarity and potential future optimizations.
    
    Args:
        token: JWT token string
    
    Returns:
        dict: Normalized user claims
        
    Raises:
        ValueError: If token validation fails
    """
    return await verify_jwt_token(token)


def has_permission(user: dict, permission: str) -> bool:
    """
    Check if user has a required permission.
    
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
    """
    Check if user has required role.
    
    Args:
        user: Verified user claims mapping.
        role: Role name to check.
    
    Returns:
        True if the role is present in user roles.
    """
    roles = user.get("roles", [])
    return role in roles

