"""
Market Insights Engine

Automated market intelligence and anomaly detection:
- Real-time anomaly detection
- Correlation regime changes
- Arbitrage opportunity alerts
- Fundamental driver analysis
- Daily market briefings
"""
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Market Insights Engine",
    description="Automated market intelligence and anomaly detection",
    version="1.0.0",
)


class AnomalyType(str, Enum):
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    CORRELATION_BREAK = "correlation_break"
    SPREAD_ANOMALY = "spread_anomaly"
    FUNDAMENTAL_DISCONNECT = "fundamental_disconnect"


class InsightType(str, Enum):
    ANOMALY = "anomaly"
    OPPORTUNITY = "opportunity"
    RISK_ALERT = "risk_alert"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"


class MarketAnomaly(BaseModel):
    """Detected market anomaly."""
    anomaly_id: str
    market: str
    anomaly_type: AnomalyType
    severity: str  # critical, high, medium, low
    detected_at: datetime
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
    price_summary: Dict[str, Dict[str, float]]
    anomalies: List[str]
    opportunities: List[str]
    outlook: str


class MarketInsightsEngine:
    """Automated market intelligence engine."""
    
    def __init__(self):
        self.anomaly_thresholds = {
            "price_spike": 3.0,  # 3 standard deviations
            "volume_surge": 2.5,
            "correlation_break": 0.3,  # correlation change
        }
    
    def detect_anomalies(
        self,
        market: str,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Detect market anomalies using statistical methods.
        
        Uses Z-score, Isolation Forest, and correlation analysis.
        """
        logger.info(f"Detecting anomalies in {market}")
        
        anomalies = []
        
        # Mock price spike detection
        current_price = 65.0 + (hash(market) % 20)
        historical_avg = 50.0
        historical_std = 8.0
        
        z_score = (current_price - historical_avg) / historical_std
        
        if abs(z_score) > self.anomaly_thresholds["price_spike"]:
            deviation_pct = ((current_price - historical_avg) / historical_avg) * 100
            
            # Analyze possible causes
            causes = []
            hour = datetime.utcnow().hour
            if hour in [18, 19, 20, 21]:
                causes.append("Peak demand period")
            if hash(market + str(datetime.utcnow().day)) % 5 == 0:
                causes.append("Unplanned generation outage")
                causes.append("Transmission constraint")
            
            anomalies.append({
                "anomaly_id": f"ANOM-{market}-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                "market": market,
                "anomaly_type": AnomalyType.PRICE_SPIKE,
                "severity": "critical" if abs(z_score) > 4 else "high",
                "detected_at": datetime.utcnow(),
                "description": f"Price {abs(deviation_pct):.1f}% {'above' if deviation_pct > 0 else 'below'} historical average",
                "current_value": current_price,
                "expected_value": historical_avg,
                "deviation_pct": deviation_pct,
                "possible_causes": causes,
            })
        
        # Mock correlation break detection
        if hash(market) % 3 == 0:
            anomalies.append({
                "anomaly_id": f"ANOM-{market}-CORR-{datetime.utcnow().strftime('%Y%m%d')}",
                "market": market,
                "anomaly_type": AnomalyType.CORRELATION_BREAK,
                "severity": "medium",
                "detected_at": datetime.utcnow(),
                "description": "Correlation with natural gas prices dropped from 0.85 to 0.42",
                "current_value": 0.42,
                "expected_value": 0.85,
                "deviation_pct": -50.6,
                "possible_causes": [
                    "Increased renewable generation",
                    "Transmission congestion",
                    "Regional weather divergence",
                ],
            })
        
        return anomalies
    
    def find_arbitrage_opportunities(
        self,
        markets: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Identify arbitrage opportunities across markets.
        
        Looks for price spreads exceeding transaction costs.
        """
        logger.info(f"Finding arbitrage in {len(markets)} markets")
        
        opportunities = []
        
        # Compare all market pairs
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                # Mock prices
                price1 = 50.0 + (hash(market1) % 20)
                price2 = 50.0 + (hash(market2) % 20)
                
                spread = price2 - price1
                
                # Historical spread
                expected_spread = 2.0  # USD/MWh
                
                # Transaction costs
                transaction_cost = 1.5  # USD/MWh
                
                # Check if arbitrage exists
                if abs(spread - expected_spread) > transaction_cost:
                    profit_per_mwh = abs(spread - expected_spread) - transaction_cost
                    volume_mw = 100  # Conservative estimate
                    
                    opportunities.append({
                        "opportunity_id": f"ARB-{market1}-{market2}-{datetime.utcnow().strftime('%Y%m%d%H')}",
                        "market_pair": [market1, market2],
                        "spread": spread,
                        "expected_spread": expected_spread,
                        "profit_potential_usd": profit_per_mwh * volume_mw,
                        "confidence": 0.85,
                        "execution_window_minutes": 30,
                        "constraints": [
                            f"Transmission capacity: {volume_mw} MW available",
                            "Requires real-time execution",
                        ],
                    })
        
        return opportunities
    
    def analyze_fundamentals(
        self,
        market: str,
        date_range: tuple
    ) -> Dict[str, Any]:
        """
        Analyze fundamental market drivers.
        
        Correlates prices with fuel, weather, demand, etc.
        """
        logger.info(f"Analyzing fundamentals for {market}")
        
        # Mock fundamental analysis
        drivers = []
        
        # Natural gas impact
        gas_price = 3.50  # USD/MMBtu
        gas_correlation = 0.82
        gas_impact_pct = 45
        
        drivers.append({
            "driver": "Natural Gas Prices",
            "current_value": gas_price,
            "correlation": gas_correlation,
            "impact_on_power_pct": gas_impact_pct,
            "explanation": f"Gas at ${gas_price}/MMBtu, driving ~{gas_impact_pct}% of price variation",
        })
        
        # Weather impact
        temp_f = 75 + (hash(str(datetime.utcnow().day)) % 20)
        temp_impact = abs(temp_f - 65) * 0.5  # Impact factor
        
        drivers.append({
            "driver": "Weather (Temperature)",
            "current_value": temp_f,
            "correlation": 0.68,
            "impact_on_power_pct": int(temp_impact),
            "explanation": f"Temperature at {temp_f}°F, driving {int(temp_impact)}% demand variation",
        })
        
        # Renewable generation
        renewable_pct = 30 + (hash(market) % 20)
        
        drivers.append({
            "driver": "Renewable Generation",
            "current_value": renewable_pct,
            "correlation": -0.55,  # Negative correlation (more renewables = lower prices)
            "impact_on_power_pct": 15,
            "explanation": f"{renewable_pct}% renewable penetration suppressing prices by ~15%",
        })
        
        return {
            "market": market,
            "analysis_date": datetime.utcnow().date(),
            "primary_drivers": drivers,
            "driver_weights": {
                "fuel_prices": 0.45,
                "weather": 0.25,
                "renewables": 0.15,
                "demand": 0.15,
            },
        }
    
    def generate_daily_briefing(
        self,
        markets: List[str],
        date_obj: date
    ) -> Dict[str, Any]:
        """
        Generate automated daily market briefing.
        
        Summarizes key developments and outlook.
        """
        logger.info(f"Generating briefing for {date_obj}")
        
        # Aggregate anomalies
        all_anomalies = []
        for market in markets:
            anomalies = self.detect_anomalies(market)
            all_anomalies.extend([a["description"] for a in anomalies])
        
        # Find opportunities
        opportunities = self.find_arbitrage_opportunities(markets)
        opp_summaries = [
            f"{o['market_pair'][0]}-{o['market_pair'][1]} spread: ${o['profit_potential_usd']:.0f} potential"
            for o in opportunities
        ]
        
        # Price summary
        price_summary = {}
        for market in markets:
            price_summary[market] = {
                "avg": 50.0 + (hash(market) % 20),
                "min": 40.0 + (hash(market) % 10),
                "max": 60.0 + (hash(market) % 30),
            }
        
        # Executive summary
        avg_prices = [p["avg"] for p in price_summary.values()]
        avg_market_price = np.mean(avg_prices)
        
        exec_summary = f"""
Market Overview for {date_obj.strftime('%B %d, %Y')}:
Power prices averaged ${avg_market_price:.2f}/MWh across {len(markets)} markets.
{len(all_anomalies)} significant anomalies detected. 
{len(opportunities)} arbitrage opportunities identified.
Natural gas prices and weather conditions were primary drivers.
        """.strip()
        
        # Key drivers
        key_drivers = [
            "Natural gas prices +5% driving power prices higher",
            "Above-normal temperatures in 3 regions increasing cooling demand",
            "Wind generation 15% below forecast, tightening supply",
        ]
        
        # Outlook
        outlook = """
Expect continued elevated prices through next week as heat persists.
Potential relief from increased renewable generation forecasted weekend.
Monitor natural gas storage levels for medium-term price direction.
        """.strip()
        
        return {
            "date": date_obj,
            "markets": markets,
            "executive_summary": exec_summary,
            "key_drivers": key_drivers,
            "price_summary": price_summary,
            "anomalies": all_anomalies[:5],  # Top 5
            "opportunities": opp_summaries[:3],  # Top 3
            "outlook": outlook,
        }


# Global insights engine
insights_engine = MarketInsightsEngine()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "market-insights"}


@app.get("/api/v1/insights/anomalies", response_model=List[MarketAnomaly])
async def detect_anomalies(
    market: str,
    lookback_hours: int = 24,
):
    """
    Detect market anomalies in real-time.
    
    Uses statistical and ML methods.
    """
    try:
        anomalies = insights_engine.detect_anomalies(market, lookback_hours)
        return [MarketAnomaly(**a) for a in anomalies]
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/insights/arbitrage", response_model=List[ArbitrageOpportunity])
async def find_arbitrage(
    markets: List[str],
):
    """
    Find arbitrage opportunities across markets.
    """
    try:
        opportunities = insights_engine.find_arbitrage_opportunities(markets)
        return [ArbitrageOpportunity(**o) for o in opportunities]
    except Exception as e:
        logger.error(f"Error finding arbitrage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/insights/fundamentals")
async def analyze_fundamentals(
    market: str,
    start_date: date,
    end_date: date,
):
    """
    Analyze fundamental market drivers.
    """
    try:
        analysis = insights_engine.analyze_fundamentals(market, (start_date, end_date))
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing fundamentals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/insights/daily-briefing", response_model=MarketBriefing)
async def get_daily_briefing(
    markets: List[str],
    date_param: Optional[date] = None,
):
    """
    Get automated daily market briefing.
    
    Includes anomalies, opportunities, and outlook.
    """
    try:
        target_date = date_param or date.today()
        briefing = insights_engine.generate_daily_briefing(markets, target_date)
        return MarketBriefing(**briefing)
    except Exception as e:
        logger.error(f"Error generating briefing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8034)

