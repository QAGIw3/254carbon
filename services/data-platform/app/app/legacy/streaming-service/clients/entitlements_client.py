"""
HTTP client for Entitlements Service.
"""
import logging
import os
from typing import List, Dict, Any

import httpx

logger = logging.getLogger(__name__)

ENTITLEMENTS_SERVICE_URL = os.getenv(
    "ENTITLEMENTS_SERVICE_URL",
    "http://entitlements-service:8011"
)


async def check_entitlement(
    user: dict,
    instrument_id: str,
    channel: str,
) -> bool:
    """
    Check user entitlement via Entitlements Service.
    
    Args:
        user: User claims dict with user_id and tenant_id
        instrument_id: Instrument ID to check
        channel: Access channel (api, downloads, stream, hub)
    
    Returns:
        bool: True if user is entitled
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{ENTITLEMENTS_SERVICE_URL}/entitlements/check",
                json={
                    "user_id": user.get("sub"),
                    "tenant_id": user.get("tenant_id"),
                    "instrument_id": instrument_id,
                    "channel": channel
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Entitlements service returned {response.status_code}")
                return False
            
            data = response.json()
            return data.get("entitled", False)
            
    except Exception as e:
        logger.error(f"Error checking entitlement: {e}")
        # Fail open in case of service issues (adjust based on security requirements)
        return False


async def bulk_check_entitlements(
    user: dict,
    checks: List[dict],
) -> List[Dict[str, Any]]:
    """
    Check multiple entitlements efficiently.
    
    Args:
        user: User claims dict
        checks: List of {instrument_id, channel} dicts
    
    Returns:
        List of {instrument_id, channel, entitled} results
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{ENTITLEMENTS_SERVICE_URL}/entitlements/bulk-check",
                json={
                    "user_id": user.get("sub"),
                    "tenant_id": user.get("tenant_id"),
                    "checks": checks
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Bulk entitlements check failed: {response.status_code}")
                return []
            
            data = response.json()
            return data.get("results", [])
            
    except Exception as e:
        logger.error(f"Error in bulk entitlements check: {e}")
        return []

