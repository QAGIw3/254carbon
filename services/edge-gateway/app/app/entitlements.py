"""
Entitlement checking logic.
"""
import logging
from typing import Optional

import sys
import os

# Add current directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from db import get_postgres_pool

logger = logging.getLogger(__name__)


async def check_entitlement(
    user: dict,
    instrument_id: str,
    channel: str,
) -> bool:
    """Check entitlement for an instrument and channel.

    Args:
        user: Verified user claims dict from auth.verify_token.
        instrument_id: Instrument ID or market/product tuple key.
        channel: One of "hub", "api", or "downloads".

    Returns:
        bool: True if the user’s tenant has the required entitlement.
    """
    tenant_id = user.get("tenant_id")

    if not tenant_id:
        logger.warning("User has no tenant_id")
        return False

    try:
        pool = await get_postgres_pool()
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
