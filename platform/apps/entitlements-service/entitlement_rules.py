"""
Entitlement business logic and rules.
"""
import logging
from typing import List, Dict, Any

from db import get_pool

logger = logging.getLogger(__name__)


async def check_user_entitlement(
    user: dict,
    instrument_id: str,
    channel: str,
) -> bool:
    """
    Check entitlement for an instrument and channel.
    
    Args:
        user: User claims dict with tenant_id.
        instrument_id: Instrument ID to check access for.
        channel: One of "hub", "api", "downloads", "stream".
    
    Returns:
        bool: True if the user's tenant has the required entitlement.
    """
    tenant_id = user.get("tenant_id")

    if not tenant_id:
        logger.warning("User has no tenant_id")
        return False

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Get instrument details
            instrument = await conn.fetchrow(
                "SELECT market, product FROM pg.instrument WHERE instrument_id = $1",
                instrument_id,
            )

            if not instrument:
                logger.warning(f"Instrument not found: {instrument_id}")
                return False

            # Check entitlement
            entitlement = await conn.fetchrow(
                """
                SELECT channels
                FROM pg.entitlement_product
                WHERE tenant_id = $1
                  AND market = $2
                  AND product = $3
                  AND (from_date IS NULL OR from_date <= CURRENT_DATE)
                  AND (to_date IS NULL OR to_date >= CURRENT_DATE)
                """,
                tenant_id,
                instrument["market"],
                instrument["product"],
            )

            if not entitlement:
                logger.info(
                    f"No entitlement for tenant {tenant_id}, "
                    f"market {instrument['market']}, product {instrument['product']}"
                )
                return False

            # Check channel access
            channels = entitlement["channels"]
            has_access = channels.get(channel, False)

            if not has_access:
                logger.info(
                    f"Tenant {tenant_id} not entitled to {channel} "
                    f"for {instrument['market']}/{instrument['product']}"
                )

            return has_access

    except Exception as e:
        logger.error(f"Error checking entitlement: {e}")
        return False


async def bulk_check_entitlements(
    user: dict,
    checks: List[dict],
) -> List[Dict[str, Any]]:
    """
    Check multiple entitlements efficiently.
    
    Args:
        user: User claims dict with tenant_id.
        checks: List of {instrument_id, channel} dicts to check.
    
    Returns:
        List of {instrument_id, channel, entitled} results.
    """
    results = []
    
    for check in checks:
        instrument_id = check.get("instrument_id")
        channel = check.get("channel")
        
        if not instrument_id or not channel:
            results.append({
                "instrument_id": instrument_id,
                "channel": channel,
                "entitled": False,
                "reason": "Missing instrument_id or channel",
            })
            continue
        
        entitled = await check_user_entitlement(user, instrument_id, channel)
        
        results.append({
            "instrument_id": instrument_id,
            "channel": channel,
            "entitled": entitled,
        })
    
    return results


async def get_user_entitlements_list(user: dict) -> List[Dict[str, Any]]:
    """
    Get all entitlements for a user's tenant.
    
    Args:
        user: User claims dict with tenant_id.
    
    Returns:
        List of entitlement records with market, product, and channels.
    """
    tenant_id = user.get("tenant_id")

    if not tenant_id:
        logger.warning("User has no tenant_id")
        return []

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT market, product, channels, from_date, to_date
                FROM pg.entitlement_product
                WHERE tenant_id = $1
                  AND (from_date IS NULL OR from_date <= CURRENT_DATE)
                  AND (to_date IS NULL OR to_date >= CURRENT_DATE)
                ORDER BY market, product
                """,
                tenant_id,
            )

            entitlements = [
                {
                    "market": row["market"],
                    "product": row["product"],
                    "channels": row["channels"],
                    "from_date": row["from_date"].isoformat() if row["from_date"] else None,
                    "to_date": row["to_date"].isoformat() if row["to_date"] else None,
                }
                for row in rows
            ]

            return entitlements

    except Exception as e:
        logger.error(f"Error fetching user entitlements: {e}")
        return []

