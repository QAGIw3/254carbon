"""
Market Insights API Router

Purpose
-------
Exposes anomaly detection, cross-market arbitrage, and daily briefings.
Uses `MarketInsightsEngine` to encapsulate the domain logic.
"""
import logging
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from insights.engine import MarketInsightsEngine, AnomalyType, InsightType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/insights", tags=["insights"])

# Initialize engine
insights_engine = MarketInsightsEngine()


class MarketAnomaly(BaseModel):
    """Detected market anomaly."""
    anomaly_id: str
    market: str
    anomaly_type: AnomalyType
    severity: str
    detected_at: str
    description: str
    current_value: float
    expected_value: float
    deviation_pct: float
    possible_causes: List[str]


class ArbitrageOpportunity(BaseModel):
    """Identified arbitrage opportunity."""
    opportunity_id: str
    market_pair: List[str]
    spread: float
    expected_spread: float
    profit_potential_usd: float
    confidence: float
    execution_window_minutes: int
    constraints: List[str]


class MarketBriefing(BaseModel):
    """Daily market briefing."""
    date: date
    markets: List[str]
    executive_summary: str
    key_drivers: List[str]
    price_summary: dict
    anomalies: List[str]
    opportunities: List[str]
    outlook: str


@router.get("/anomalies", response_model=List[MarketAnomaly])
async def detect_anomalies(
    market: str,
    lookback_hours: int = 24,
):
    """Detect market anomalies in real-time."""
    try:
        anomalies = insights_engine.detect_anomalies(market, lookback_hours)
        # Convert datetime to string for pydantic
        for a in anomalies:
            a["detected_at"] = a["detected_at"].isoformat()
        return [MarketAnomaly(**a) for a in anomalies]
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/arbitrage", response_model=List[ArbitrageOpportunity])
async def find_arbitrage(
    markets: List[str],
):
    """Find arbitrage opportunities across markets."""
    try:
        opportunities = insights_engine.find_arbitrage_opportunities(markets)
        return [ArbitrageOpportunity(**o) for o in opportunities]
    except Exception as e:
        logger.error(f"Error finding arbitrage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fundamentals")
async def analyze_fundamentals(
    market: str,
    start_date: date,
    end_date: date,
):
    """Analyze fundamental market drivers."""
    try:
        analysis = insights_engine.analyze_fundamentals(market, (start_date, end_date))
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing fundamentals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily-briefing", response_model=MarketBriefing)
async def get_daily_briefing(
    markets: List[str],
    date_param: Optional[date] = None,
):
    """Get automated daily market briefing."""
    try:
        target_date = date_param or date.today()
        briefing = insights_engine.generate_daily_briefing(markets, target_date)
        return MarketBriefing(**briefing)
    except Exception as e:
        logger.error(f"Error generating briefing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
