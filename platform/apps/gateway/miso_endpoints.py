"""
MISO-specific API endpoints for pilot customer workflows.

These endpoints provide customized functionality for MISO trading workflows,
including position summaries, daily reports, portfolio risk, price alerts,
congestion analysis, and trading opportunities. Entitlements are enforced
to ensure users have access to the MISO market API channel.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from auth import verify_token
from entitlements import check_entitlement
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import db
from metrics import track_request

logger = logging.getLogger(__name__)

# Create MISO-specific router for namespacing and OpenAPI grouping
miso_router = APIRouter(prefix="/api/v1/miso", tags=["MISO"])


# Models for MISO-specific endpoints

class MISOPositionSummary(BaseModel):
    """MISO position summary for trading desks."""
    date: str
    total_positions: int
    total_volume_mw: float
    avg_price: float
    pnl_unrealized: float
    pnl_realized: float
    market_positions: Dict[str, Dict[str, Any]]


class MISODailyTradingReport(BaseModel):
    """Daily trading report for MISO operations."""
    report_date: str
    total_volume_mw: float
    total_revenue: float
    avg_price: float
    peak_price: float
    lowest_price: float
    price_volatility: float
    top_performing_nodes: List[Dict[str, Any]]
    congestion_events: int
    trading_summary: Dict[str, Any]


class MISOPortfolioRisk(BaseModel):
    """Risk metrics for MISO trading portfolio."""
    as_of_date: str
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    beta_to_hub: float
    portfolio_concentration: Dict[str, float]


class MISOPriceAlert(BaseModel):
    """Price alert configuration for MISO nodes."""
    node_id: str
    alert_type: str  # "price_threshold", "volatility", "spike"
    threshold: float
    direction: str  # "above", "below", "both"
    frequency: str  # "realtime", "hourly", "daily"
    enabled: bool
    last_triggered: Optional[str]


# MISO-specific endpoints

@miso_router.get("/trading-summary", response_model=MISOPositionSummary)
async def get_miso_trading_summary(
    date: str = Query(default=None, description="Date in YYYY-MM-DD format"),
    user=Depends(verify_token),
):
    """
    Get MISO trading position summary for a specific date.

    Auth/Entitlements
    - Requires valid token and entitlement to market=power, channel=api

    Data Source
    - PostgreSQL table miso_trading_positions (aggregated here)
    """
    track_request("get_miso_trading_summary")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    if not date:
        date = datetime.utcnow().date().isoformat()

    try:
        # Get position data from PostgreSQL
        pool = await db.get_postgres_pool()
        async with pool.acquire() as conn:
            # This would query actual trading positions
            # For now, return mock data structure
            summary = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_positions,
                    SUM(volume_mw) as total_volume,
                    AVG(price) as avg_price,
                    SUM(pnl_unrealized) as pnl_unrealized,
                    SUM(pnl_realized) as pnl_realized
                FROM miso_trading_positions
                WHERE position_date = $1
            """, date)

            return MISOPositionSummary(
                date=date,
                total_positions=summary['total_positions'] or 0,
                total_volume_mw=summary['total_volume'] or 0.0,
                avg_price=summary['avg_price'] or 0.0,
                pnl_unrealized=summary['pnl_unrealized'] or 0.0,
                pnl_realized=summary['pnl_realized'] or 0.0,
                market_positions={}
            )

    except Exception as e:
        logger.error(f"Error fetching MISO trading summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@miso_router.get("/daily-report", response_model=MISODailyTradingReport)
async def get_miso_daily_report(
    report_date: str = Query(default=None, description="Report date in YYYY-MM-DD format"),
    user=Depends(verify_token),
):
    """
    Get comprehensive daily trading report for MISO.

    Data Sourcing
    - ClickHouse: ch.market_price_ticks for summary metrics
    - Top nodes are mocked placeholders pending full PnL calculation
    """
    track_request("get_miso_daily_report")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    if not report_date:
        report_date = (datetime.utcnow() - timedelta(days=1)).date().isoformat()

    try:
        # Get trading data from ClickHouse
        clickhouse = await db.get_clickhouse_client()

        # Get daily trading metrics
        query = """
            SELECT
                market,
                SUM(volume) as total_volume,
                AVG(value) as avg_price,
                MAX(value) as peak_price,
                MIN(value) as lowest_price,
                STDDEV(value) as price_volatility
            FROM ch.market_price_ticks
            WHERE market = 'MISO'
                AND event_time >= toDateTime('{report_date}')
                AND event_time < toDateTime('{report_date}') + INTERVAL 1 DAY
            GROUP BY market
        """

        result = await clickhouse.fetch_one(query)

        if not result:
            raise HTTPException(status_code=404, detail="No data found for the specified date")

        # Get top performing nodes (mock data for now)
        top_nodes = [
            {"node_id": "MISO.INDIANA_HUB", "pnl": 15000.0, "volume": 100.0},
            {"node_id": "MISO.MICHIGAN_HUB", "pnl": 12000.0, "volume": 80.0},
            {"node_id": "MISO.MINNESOTA_HUB", "pnl": 8000.0, "volume": 60.0},
        ]

        return MISODailyTradingReport(
            report_date=report_date,
            total_volume_mw=result['total_volume'] or 0.0,
            total_revenue=result['total_volume'] * result['avg_price'] or 0.0,
            avg_price=result['avg_price'] or 0.0,
            peak_price=result['peak_price'] or 0.0,
            lowest_price=result['lowest_price'] or 0.0,
            price_volatility=result['price_volatility'] or 0.0,
            top_performing_nodes=top_nodes,
            congestion_events=0,  # Would need to calculate from congestion data
            trading_summary={
                "total_trades": 150,
                "successful_trades": 135,
                "success_rate": 90.0
            }
        )

    except Exception as e:
        logger.error(f"Error fetching MISO daily report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@miso_router.get("/portfolio-risk", response_model=MISOPortfolioRisk)
async def get_miso_portfolio_risk(
    as_of_date: str = Query(default=None, description="As-of date in YYYY-MM-DD format"),
    user=Depends(verify_token),
):
    """
    Get risk metrics for MISO trading portfolio.

    Notes
    - Mock calculation stub; integrate with portfolio risk engine in prod
    """
    track_request("get_miso_portfolio_risk")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    if not as_of_date:
        as_of_date = datetime.utcnow().date().isoformat()

    try:
        # Calculate risk metrics (mock implementation)
        # In reality, this would use sophisticated risk models

        risk_metrics = MISOPortfolioRisk(
            as_of_date=as_of_date,
            var_95=250000.0,  # 95% Value at Risk
            var_99=450000.0,  # 99% Value at Risk
            expected_shortfall=380000.0,  # Expected Shortfall
            max_drawdown=150000.0,  # Maximum Drawdown
            sharpe_ratio=1.8,  # Sharpe Ratio
            beta_to_hub=0.85,  # Beta to Indiana Hub
            portfolio_concentration={
                "INDIANA_HUB": 45.0,
                "MICHIGAN_HUB": 30.0,
                "MINNESOTA_HUB": 15.0,
                "OTHER_NODES": 10.0
            }
        )

        return risk_metrics

    except Exception as e:
        logger.error(f"Error calculating MISO portfolio risk: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@miso_router.get("/price-alerts", response_model=List[MISOPriceAlert])
async def get_miso_price_alerts(user=Depends(verify_token)):
    """Get configured price alerts for MISO nodes for the current user."""
    track_request("get_miso_price_alerts")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        # Get configured alerts from PostgreSQL
        pool = await db.get_postgres_pool()
        async with pool.acquire() as conn:
            alerts = await conn.fetch("""
                SELECT node_id, alert_type, threshold, direction,
                       frequency, enabled, last_triggered
                FROM miso_price_alerts
                WHERE user_id = $1
                ORDER BY node_id
            """, user.get("sub"))

            return [
                MISOPriceAlert(
                    node_id=row['node_id'],
                    alert_type=row['alert_type'],
                    threshold=row['threshold'],
                    direction=row['direction'],
                    frequency=row['frequency'],
                    enabled=row['enabled'],
                    last_triggered=row['last_triggered']
                )
                for row in alerts
            ]

    except Exception as e:
        logger.error(f"Error fetching MISO price alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@miso_router.post("/price-alerts")
async def create_miso_price_alert(
    alert: MISOPriceAlert,
    user=Depends(verify_token),
):
    """Create a new price alert for MISO nodes."""
    track_request("create_miso_price_alert")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        pool = await db.get_postgres_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO miso_price_alerts
                (user_id, node_id, alert_type, threshold, direction, frequency, enabled, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            """, user.get("sub"), alert.node_id, alert.alert_type,
                 alert.threshold, alert.direction, alert.frequency, alert.enabled)

            return {"status": "created", "alert": alert.dict()}

    except Exception as e:
        logger.error(f"Error creating MISO price alert: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@miso_router.get("/congestion-analysis")
async def get_miso_congestion_analysis(
    start_date: str = Query(...),
    end_date: str = Query(...),
    user=Depends(verify_token),
):
    """
    Get MISO congestion analysis for trading decisions.

    Returns aggregates by date/flowgate including max/avg congestion price
    and rough event counts for the window.
    """
    track_request("get_miso_congestion_analysis")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        # Get congestion data from ClickHouse
        clickhouse = await db.get_clickhouse_client()

        query = """
            SELECT
                date,
                flowgate_id,
                max(congestion_price) as max_congestion,
                avg(congestion_price) as avg_congestion,
                sum(congestion_volume) as total_congestion_volume,
                count(*) as congestion_events
            FROM miso_congestion_data
            WHERE date >= '{start_date}'
                AND date <= '{end_date}'
            GROUP BY date, flowgate_id
            ORDER BY date, max_congestion DESC
        """

        results = await clickhouse.fetch_all(query)

        return {
            "period": f"{start_date} to {end_date}",
            "congestion_events": [
                {
                    "date": row['date'].isoformat(),
                    "flowgate_id": row['flowgate_id'],
                    "max_congestion": float(row['max_congestion']),
                    "avg_congestion": float(row['avg_congestion']),
                    "total_volume": float(row['total_congestion_volume']),
                    "event_count": row['congestion_events']
                }
                for row in results
            ],
            "summary": {
                "total_events": sum(row['congestion_events'] for row in results),
                "max_congestion_price": max((row['max_congestion'] for row in results), default=0),
                "avg_congestion_price": sum(row['max_congestion'] for row in results) / len(results) if results else 0
            }
        }

    except Exception as e:
        logger.error(f"Error fetching MISO congestion analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@miso_router.get("/trading-opportunities")
async def get_miso_trading_opportunities(
    min_potential_profit: float = Query(1000.0, description="Minimum potential profit threshold"),
    user=Depends(verify_token),
):
    """
    Get MISO trading opportunities based on current market conditions.

    Currently returns mocked opportunities with indicative profitability
    and confidence to demonstrate API shape and UI integration.
    """
    track_request("get_miso_trading_opportunities")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        # This would analyze current market conditions and identify opportunities
        # For now, return mock opportunities

        opportunities = [
            {
                "opportunity_id": "opp_001",
                "type": "arbitrage",
                "description": "Price differential between Indiana Hub and Michigan Hub",
                "potential_profit": 2500.0,
                "confidence": 0.85,
                "timeframe": "next_2_hours",
                "nodes_involved": ["MISO.INDIANA_HUB", "MISO.MICHIGAN_HUB"],
                "recommended_action": "Buy Indiana Hub, Sell Michigan Hub",
                "risk_level": "low"
            },
            {
                "opportunity_id": "opp_002",
                "type": "congestion_trade",
                "description": "Congestion expected on MN-WI flowgate",
                "potential_profit": 1800.0,
                "confidence": 0.72,
                "timeframe": "next_4_hours",
                "nodes_involved": ["MISO.MINNESOTA_HUB", "MISO.WISCONSIN_HUB"],
                "recommended_action": "Position for congestion pricing",
                "risk_level": "medium"
            }
        ]

        return {
            "opportunities": opportunities,
            "total_count": len(opportunities),
            "generated_at": datetime.utcnow().isoformat(),
            "parameters": {
                "min_profit_threshold": min_potential_profit,
                "analysis_window": "2_hours"
            }
        }

    except Exception as e:
        logger.error(f"Error analyzing MISO trading opportunities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
