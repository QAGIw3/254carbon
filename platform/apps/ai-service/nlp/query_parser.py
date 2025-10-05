"""
NLP Query Parser

Overview
--------
Parses natural language questions into structured intents, entities, and a
plan of action (SQL or API calls) along with a light-weight narrative.

Implementation Notes
--------------------
- Intent/entity extraction is heuristic for development and tests.
- Replace with a proper NLP pipeline or LLM function-calling in production.
"""
import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class NLPEngine:
    """Natural language processing engine.

    Responsibilities
    ----------------
    - Classify intent (data query, comparison, prediction, general)
    - Extract key entities (market, metric, dates)
    - Build SQL or downstream API call hints
    - Generate a brief narrative for UI display
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    async def parse_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Parse natural language query into structured format.

        Returns a dict with: intent, entities, optional SQL/API plan, and
        a short narrative suitable for display.
        """
        intent = self._classify_intent(query)
        entities = self._extract_entities(query)
        
        if intent == "data_query":
            sql_query = self._generate_sql(query, entities)
            api_calls = None
        elif intent == "comparison":
            sql_query = None
            api_calls = self._generate_api_calls(query, entities)
        else:
            sql_query = None
            api_calls = None
        
        narrative = await self._generate_narrative(query, entities, context)
        
        return {
            "intent": intent,
            "entities": entities,
            "sql_query": sql_query,
            "api_calls": api_calls,
            "narrative_response": narrative,
        }
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent using simple lexical cues."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "show", "get"]):
            return "data_query"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["forecast", "predict"]):
            return "prediction"
        else:
            return "general"
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query (market, dates, metric)."""
        entities = {}
        query_lower = query.lower()
        
        markets = ["pjm", "miso", "caiso", "ercot", "spp", "nyiso"]
        for market in markets:
            if market in query_lower:
                entities["market"] = market.upper()
                break
        
        if "yesterday" in query_lower:
            entities["start_date"] = (datetime.utcnow() - timedelta(days=1)).date()
            entities["end_date"] = entities["start_date"]
        
        if "price" in query_lower:
            entities["metric"] = "price"
        elif "congestion" in query_lower:
            entities["metric"] = "congestion"
        
        return entities
    
    def _generate_sql(self, query: str, entities: Dict[str, Any]) -> str:
        """Generate SQL query from natural language using simple templates."""
        market = entities.get("market", "PJM")
        metric = entities.get("metric", "price")
        
        sql = f"""
        SELECT 
            DATE(event_time_utc) as date,
            AVG(value) as {metric}_avg
        FROM ch.market_price_ticks
        WHERE instrument_id LIKE '{market}%'
        """
        
        if "start_date" in entities:
            sql += f"\n  AND event_time_utc >= '{entities['start_date']}'"
        
        sql += "\nGROUP BY date\nORDER BY date"
        return sql
    
    def _generate_api_calls(self, query: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate API calls from natural language (suggested downstream calls)."""
        return [{
            "endpoint": "/api/v1/prices/historical",
            "method": "GET",
            "params": {
                "market": entities.get("market"),
                "start_date": entities.get("start_date"),
            }
        }]
    
    async def _generate_narrative(
        self, query: str, entities: Dict[str, Any], context: Optional[Dict] = None
    ) -> str:
        """Generate narrative response (LLM placeholder)."""
        market = entities.get("market", "the market")
        metric = entities.get("metric", "prices")
        return f"I'll analyze {metric} in {market} for you."
    
    async def generate_insights(
        self, market: str, timeframe: str, data_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate automated market insights (light-weight stub)."""
        return {
            "title": f"{market} Market Update - {timeframe.title()}",
            "summary": f"Analysis of {market} power market conditions",
            "key_findings": [
                f"Average prices in {market} were stable",
                "Peak demand occurred during evening hours",
                "Renewable generation increased",
            ],
            "detailed_analysis": f"The {market} market showed typical patterns during {timeframe}.",
        }
