"""
Real-time price alert system for MISO customers.
Monitors price movements and sends alerts when thresholds are exceeded.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

from fastapi import BackgroundTasks

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import db
from cache import CacheStrategy

logger = logging.getLogger(__name__)


@dataclass
class PriceAlert:
    """Price alert configuration."""
    alert_id: str
    user_id: str
    node_id: str
    alert_type: str  # "price_threshold", "volatility", "spike"
    threshold: float
    direction: str  # "above", "below", "both"
    frequency: str  # "realtime", "hourly", "daily"
    enabled: bool
    last_triggered: Optional[datetime]
    cooldown_minutes: int = 15


class AlertManager:
    """Manage price alerts for MISO customers."""

    def __init__(self):
        self.active_alerts: Dict[str, PriceAlert] = {}
        self.alert_history: List[Dict] = []
        self.max_history_size = 1000

    async def load_alerts_from_db(self):
        """Load active alerts from database."""
        try:
            pool = await db.get_postgres_pool()
            async with pool.acquire() as conn:
                alerts = await conn.fetch("""
                    SELECT alert_id, user_id, node_id, alert_type, threshold,
                           direction, frequency, enabled, last_triggered, cooldown_minutes
                    FROM miso_price_alerts
                    WHERE enabled = true
                """)

                for alert in alerts:
                    price_alert = PriceAlert(
                        alert_id=alert['alert_id'],
                        user_id=alert['user_id'],
                        node_id=alert['node_id'],
                        alert_type=alert['alert_type'],
                        threshold=alert['threshold'],
                        direction=alert['direction'],
                        frequency=alert['frequency'],
                        enabled=alert['enabled'],
                        last_triggered=alert['last_triggered'],
                        cooldown_minutes=alert['cooldown_minutes'] or 15
                    )
                    self.active_alerts[alert['alert_id']] = price_alert

                logger.info(f"Loaded {len(alerts)} active price alerts")

        except Exception as e:
            logger.error(f"Error loading alerts from database: {e}")

    async def check_price_alerts(self):
        """Check all active price alerts against current market data."""
        if not self.active_alerts:
            await self.load_alerts_from_db()

        try:
            clickhouse = await db.get_clickhouse_client()

            # Get recent price data for all monitored nodes
            recent_prices_query = """
                SELECT
                    instrument_id,
                    value as price,
                    event_time,
                    STDDEV(value) OVER (
                        PARTITION BY instrument_id
                        ORDER BY event_time
                        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
                    ) as price_volatility
                FROM ch.market_price_ticks
                WHERE market = 'MISO'
                    AND event_time >= now() - INTERVAL 5 MINUTE
                ORDER BY instrument_id, event_time DESC
            """

            recent_prices = await clickhouse.fetch_all(recent_prices_query)

            # Group by instrument_id for easier processing
            price_data = {}
            for row in recent_prices:
                if row['instrument_id'] not in price_data:
                    price_data[row['instrument_id']] = []
                price_data[row['instrument_id']].append({
                    'price': row['price'],
                    'timestamp': row['event_time'],
                    'volatility': row['price_volatility']
                })

            # Check each alert
            triggered_alerts = []
            for alert in self.active_alerts.values():
                if not alert.enabled:
                    continue

                # Check cooldown period
                if (alert.last_triggered and
                    (datetime.utcnow() - alert.last_triggered).total_seconds() < alert.cooldown_minutes * 60):
                    continue

                # Get current price data for this node
                node_prices = price_data.get(alert.node_id, [])

                if not node_prices:
                    continue

                current_price = node_prices[0]['price']
                current_volatility = node_prices[0]['volatility']

                triggered = False
                alert_reason = ""

                if alert.alert_type == "price_threshold":
                    if alert.direction == "above" and current_price > alert.threshold:
                        triggered = True
                        alert_reason = f"Price ${current_price:.2f} exceeded threshold ${alert.threshold:.2f}"
                    elif alert.direction == "below" and current_price < alert.threshold:
                        triggered = True
                        alert_reason = f"Price ${current_price:.2f} fell below threshold ${alert.threshold:.2f}"
                    elif alert.direction == "both":
                        if current_price > alert.threshold or current_price < alert.threshold:
                            triggered = True
                            alert_reason = f"Price ${current_price:.2f} outside threshold ${alert.threshold:.2f}"

                elif alert.alert_type == "volatility":
                    # Alert if volatility exceeds threshold
                    if current_volatility > alert.threshold:
                        triggered = True
                        alert_reason = f"Price volatility {current_volatility:.2f} exceeded threshold {alert.threshold:.2f}"

                elif alert.alert_type == "spike":
                    # Check for price spikes (sudden large changes)
                    if len(node_prices) >= 2:
                        price_change = abs(node_prices[0]['price'] - node_prices[1]['price'])
                        if price_change > alert.threshold:
                            triggered = True
                            alert_reason = f"Price spike detected: ${price_change:.2f} change"

                if triggered:
                    alert_trigger = {
                        "alert_id": alert.alert_id,
                        "user_id": alert.user_id,
                        "node_id": alert.node_id,
                        "alert_type": alert.alert_type,
                        "threshold": alert.threshold,
                        "current_price": current_price,
                        "current_volatility": current_volatility,
                        "reason": alert_reason,
                        "timestamp": datetime.utcnow().isoformat()
                    }

                    triggered_alerts.append(alert_trigger)

                    # Update last triggered time
                    alert.last_triggered = datetime.utcnow()

                    # Store in history
                    self.alert_history.append(alert_trigger)
                    if len(self.alert_history) > self.max_history_size:
                        self.alert_history = self.alert_history[-self.max_history_size:]

            # Send triggered alerts
            if triggered_alerts:
                await self.send_alerts(triggered_alerts)

        except Exception as e:
            logger.error(f"Error checking price alerts: {e}")

    async def send_alerts(self, triggered_alerts: List[Dict]):
        """Send triggered alerts via email/notification system."""
        try:
            for alert in triggered_alerts:
                # Here you would integrate with your notification system
                # (email, SMS, Slack, etc.)

                logger.info(f"Price alert triggered: {alert['reason']} for node {alert['node_id']}")

                # Store alert in database
                pool = await db.get_postgres_pool()
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO miso_alert_history
                        (alert_id, user_id, node_id, alert_type, threshold,
                         current_price, reason, triggered_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    """, alert['alert_id'], alert['user_id'], alert['node_id'],
                         alert['alert_type'], alert['threshold'], alert['current_price'],
                         alert['reason'])

                # Update alert's last triggered time in database
                await conn.execute("""
                    UPDATE miso_price_alerts
                    SET last_triggered = NOW()
                    WHERE alert_id = $1
                """, alert['alert_id'])

        except Exception as e:
            logger.error(f"Error sending alerts: {e}")

    async def get_alert_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get alert history for a user."""
        try:
            pool = await db.get_postgres_pool()
            async with pool.acquire() as conn:
                history = await conn.fetch("""
                    SELECT alert_id, node_id, alert_type, threshold,
                           current_price, reason, triggered_at
                    FROM miso_alert_history
                    WHERE user_id = $1
                    ORDER BY triggered_at DESC
                    LIMIT $2
                """, user_id, limit)

                return [dict(row) for row in history]

        except Exception as e:
            logger.error(f"Error fetching alert history: {e}")
            return []


# Global alert manager instance
alert_manager = AlertManager()


# Background task for alert monitoring
async def run_alert_monitoring():
    """Run continuous price alert monitoring (to be called by scheduler)."""
    logger.info("Starting price alert monitoring...")
    while True:
        try:
            await alert_manager.check_price_alerts()
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in alert monitoring: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying


# API endpoints for alert management

@app.get("/api/v1/miso/alerts/history")
async def get_alert_history(
    limit: int = Query(50, description="Number of alerts to return"),
    user=Depends(verify_token),
):
    """Get price alert history for the current user."""
    track_request("get_alert_history")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        history = await alert_manager.get_alert_history(user.get("sub"), limit)

        return {
            "alerts": history,
            "total_count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching alert history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/miso/alerts/test")
async def test_price_alert(
    node_id: str = Query(..., description="Node ID to test alert for"),
    user=Depends(verify_token),
):
    """Test price alert for a specific node (triggers alert regardless of threshold)."""
    track_request("test_price_alert")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        # Get current price for the node
        clickhouse = await get_clickhouse_client()

        query = """
            SELECT value as price, event_time
            FROM ch.market_price_ticks
            WHERE market = 'MISO' AND instrument_id = $1
            ORDER BY event_time DESC
            LIMIT 1
        """

        result = await clickhouse.fetch_one(query, node_id)

        if not result:
            raise HTTPException(status_code=404, detail="No price data found for node")

        # Create test alert
        test_alert = {
            "alert_id": f"test_{datetime.utcnow().timestamp()}",
            "user_id": user.get("sub"),
            "node_id": node_id,
            "alert_type": "test",
            "threshold": result['price'],
            "current_price": result['price'],
            "reason": f"Test alert for node {node_id} at price ${result['price']:.2f}",
            "timestamp": datetime.utcnow().isoformat()
        }

        # Send test alert
        await alert_manager.send_alerts([test_alert])

        return {
            "status": "test_sent",
            "alert": test_alert,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error testing price alert: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/miso/alerts/status")
async def get_alert_status(user=Depends(verify_token)):
    """Get status of price alert system."""
    track_request("get_alert_status")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        return {
            "status": "active",
            "active_alerts": len(alert_manager.active_alerts),
            "total_history": len(alert_manager.alert_history),
            "last_check": datetime.utcnow().isoformat(),
            "monitoring_interval": "60 seconds"
        }

    except Exception as e:
        logger.error(f"Error getting alert status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
