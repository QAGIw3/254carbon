"""
Market Insights Engine

Automated market intelligence and anomaly detection.
"""
import logging
from datetime import datetime, date
from typing import List, Dict, Any
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


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
        """Detect market anomalies using statistical methods."""
        logger.info(f"Detecting anomalies in {market}")
        
        anomalies = []
        
        # Mock price spike detection
        current_price = 65.0 + (hash(market) % 20)
        historical_avg = 50.0
        historical_std = 8.0
        
        z_score = (current_price - historical_avg) / historical_std
        
        if abs(z_score) > self.anomaly_thresholds["price_spike"]:
            deviation_pct = ((current_price - historical_avg) / historical_avg) * 100
            
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
        """Identify arbitrage opportunities across markets."""
        logger.info(f"Finding arbitrage in {len(markets)} markets")
        
        opportunities = []
        
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                price1 = 50.0 + (hash(market1) % 20)
                price2 = 50.0 + (hash(market2) % 20)
                
                spread = price2 - price1
                expected_spread = 2.0
                transaction_cost = 1.5
                
                if abs(spread - expected_spread) > transaction_cost:
                    profit_per_mwh = abs(spread - expected_spread) - transaction_cost
                    volume_mw = 100
                    
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
        """Analyze fundamental market drivers."""
        logger.info(f"Analyzing fundamentals for {market}")
        
        drivers = []
        
        gas_price = 3.50
        gas_correlation = 0.82
        gas_impact_pct = 45
        
        drivers.append({
            "driver": "Natural Gas Prices",
            "current_value": gas_price,
            "correlation": gas_correlation,
            "impact_on_power_pct": gas_impact_pct,
            "explanation": f"Gas at ${gas_price}/MMBtu, driving ~{gas_impact_pct}% of price variation",
        })
        
        temp_f = 75 + (hash(str(datetime.utcnow().day)) % 20)
        temp_impact = abs(temp_f - 65) * 0.5
        
        drivers.append({
            "driver": "Weather (Temperature)",
            "current_value": temp_f,
            "correlation": 0.68,
            "impact_on_power_pct": int(temp_impact),
            "explanation": f"Temperature at {temp_f}°F, driving {int(temp_impact)}% demand variation",
        })
        
        renewable_pct = 30 + (hash(market) % 20)
        
        drivers.append({
            "driver": "Renewable Generation",
            "current_value": renewable_pct,
            "correlation": -0.55,
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
        """Generate automated daily market briefing."""
        logger.info(f"Generating briefing for {date_obj}")
        
        all_anomalies = []
        for market in markets:
            anomalies = self.detect_anomalies(market)
            all_anomalies.extend([a["description"] for a in anomalies])
        
        opportunities = self.find_arbitrage_opportunities(markets)
        opp_summaries = [
            f"{o['market_pair'][0]}-{o['market_pair'][1]} spread: ${o['profit_potential_usd']:.0f} potential"
            for o in opportunities
        ]
        
        price_summary = {}
        for market in markets:
            price_summary[market] = {
                "avg": 50.0 + (hash(market) % 20),
                "min": 40.0 + (hash(market) % 10),
                "max": 60.0 + (hash(market) % 30),
            }
        
        avg_prices = [p["avg"] for p in price_summary.values()]
        avg_market_price = np.mean(avg_prices)
        
        exec_summary = f"""
Market Overview for {date_obj.strftime('%B %d, %Y')}:
Power prices averaged ${avg_market_price:.2f}/MWh across {len(markets)} markets.
{len(all_anomalies)} significant anomalies detected. 
{len(opportunities)} arbitrage opportunities identified.
Natural gas prices and weather conditions were primary drivers.
        """.strip()
        
        key_drivers = [
            "Natural gas prices +5% driving power prices higher",
            "Above-normal temperatures in 3 regions increasing cooling demand",
            "Wind generation 15% below forecast, tightening supply",
        ]
        
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
            "anomalies": all_anomalies[:5],
            "opportunities": opp_summaries[:3],
            "outlook": outlook,
        }

