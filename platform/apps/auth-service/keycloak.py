"""
Keycloak integration for JWKS fetching and caching.
"""
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

# Keycloak configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080/auth/realms/254carbon")
KEYCLOAK_AUDIENCE = os.getenv("KEYCLOAK_AUDIENCE", "market-intelligence-api")

# Cache for JWKS (JSON Web Key Set)
_jwks_cache: Optional[Dict[str, Any]] = None
_jwks_cache_time: Optional[datetime] = None
_CACHE_DURATION = timedelta(hours=1)


async def get_jwks() -> Dict[str, Any]:
    """
    Fetch JWKS from Keycloak with caching.
    
    JWKS contains public keys used to verify JWT signatures.
    Cached for 1 hour to reduce Keycloak load.
    
    Returns:
        Dict[str, Any]: JWKS payload from authorization server.
        
    Raises:
        Exception: If Keycloak is unreachable or returns invalid data.
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

    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching JWKS: {e}")
        raise Exception("Keycloak unreachable")
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise Exception("Authentication service unavailable")


def clear_jwks_cache():
    """Clear the JWKS cache to force refresh."""
    global _jwks_cache, _jwks_cache_time
    _jwks_cache = None
    _jwks_cache_time = None
    logger.info("JWKS cache cleared")


def find_key_by_kid(kid: str, jwks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find the key with matching kid in JWKS.
    
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

