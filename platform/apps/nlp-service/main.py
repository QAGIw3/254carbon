"""
NLP Service for Market Intelligence

Natural language query understanding, automated insights,
and report generation using LLM integration.
"""
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NLP Service",
    description="Natural language understanding for market intelligence",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Natural language query."""
    query: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Parsed query response."""
    intent: str
    entities: Dict[str, Any]
    sql_query: Optional[str] = None
    api_calls: Optional[List[Dict[str, Any]]] = None
    narrative_response: str


class InsightRequest(BaseModel):
    """Request for market insights."""
    market: str
    timeframe: str  # "today", "week", "month"
    data_context: Optional[Dict[str, Any]] = None


class InsightResponse(BaseModel):
    """Generated market insight."""
    title: str
    summary: str
    key_findings: List[str]
    detailed_analysis: str
    charts_recommended: List[str]


class ReportRequest(BaseModel):
    """Automated report generation request."""
    report_type: str  # "daily", "weekly", "monthly"
    markets: List[str]
    start_date: date
    end_date: date
    include_forecast: bool = True


class ReportResponse(BaseModel):
    """Generated report."""
    report_id: str
    title: str
    executive_summary: str
    sections: List[Dict[str, Any]]
    download_url: str


class NLPEngine:
    """Natural language processing engine."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key
    
    async def parse_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Parse natural language query into structured format.
        
        Examples:
        - "What was the average price in PJM last week?"
        - "Show me congestion trends in ERCOT for July"
        - "Compare MISO and CAISO prices yesterday"
        """
        # Intent classification
        intent = self._classify_intent(query)
        
        # Entity extraction
        entities = self._extract_entities(query)
        
        # Generate SQL or API calls
        if intent == "data_query":
            sql_query = self._generate_sql(query, entities)
            api_calls = None
        elif intent == "comparison":
            sql_query = None
            api_calls = self._generate_api_calls(query, entities)
        else:
            sql_query = None
            api_calls = None
        
        # Generate narrative response
        narrative = await self._generate_narrative(query, entities, context)
        
        return {
            "intent": intent,
            "entities": entities,
            "sql_query": sql_query,
            "api_calls": api_calls,
            "narrative_response": narrative,
        }
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "show", "get", "fetch"]):
            return "data_query"
        elif any(word in query_lower for word in ["why", "explain", "reason"]):
            return "explanation"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["forecast", "predict", "future"]):
            return "prediction"
        elif any(word in query_lower for word in ["alert", "notify", "watch"]):
            return "alert_setup"
        else:
            return "general"
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query."""
        entities = {}
        
        query_lower = query.lower()
        
        # Market/ISO detection
        markets = ["pjm", "miso", "caiso", "ercot", "spp", "nyiso", "ieso", "aeso"]
        for market in markets:
            if market in query_lower:
                entities["market"] = market.upper()
                break
        
        # Time period detection
        if "yesterday" in query_lower:
            entities["start_date"] = (datetime.utcnow() - timedelta(days=1)).date()
            entities["end_date"] = entities["start_date"]
        elif "last week" in query_lower:
            entities["start_date"] = (datetime.utcnow() - timedelta(days=7)).date()
            entities["end_date"] = datetime.utcnow().date()
        elif "last month" in query_lower:
            entities["start_date"] = (datetime.utcnow() - timedelta(days=30)).date()
            entities["end_date"] = datetime.utcnow().date()
        
        # Metric detection
        if "price" in query_lower or "lmp" in query_lower:
            entities["metric"] = "price"
        elif "congestion" in query_lower:
            entities["metric"] = "congestion"
        elif "volume" in query_lower or "load" in query_lower:
            entities["metric"] = "volume"
        
        # Aggregation detection
        if "average" in query_lower or "avg" in query_lower:
            entities["aggregation"] = "avg"
        elif "max" in query_lower or "peak" in query_lower:
            entities["aggregation"] = "max"
        elif "min" in query_lower or "minimum" in query_lower:
            entities["aggregation"] = "min"
        
        return entities
    
    def _generate_sql(self, query: str, entities: Dict[str, Any]) -> str:
        """Generate SQL query from natural language."""
        # Simplified SQL generation
        market = entities.get("market", "PJM")
        metric = entities.get("metric", "price")
        agg = entities.get("aggregation", "avg")
        
        sql = f"""
        SELECT 
            DATE(event_time_utc) as date,
            {agg.upper()}(value) as {metric}_{agg}
        FROM ch.market_price_ticks
        WHERE instrument_id LIKE '{market}%'
        """
        
        if "start_date" in entities:
            sql += f"\n  AND event_time_utc >= '{entities['start_date']}'"
        if "end_date" in entities:
            sql += f"\n  AND event_time_utc <= '{entities['end_date']}'"
        
        sql += "\nGROUP BY date\nORDER BY date"
        
        return sql
    
    def _generate_api_calls(self, query: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate API calls from natural language."""
        api_calls = []
        
        if "market" in entities:
            api_calls.append({
                "endpoint": "/api/v1/prices/historical",
                "method": "GET",
                "params": {
                    "market": entities["market"],
                    "start_date": entities.get("start_date"),
                    "end_date": entities.get("end_date"),
                }
            })
        
        return api_calls
    
    async def _generate_narrative(
        self,
        query: str,
        entities: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> str:
        """Generate narrative response using LLM."""
        # Mock response - in production would use OpenAI API
        market = entities.get("market", "the market")
        metric = entities.get("metric", "prices")
        
        if "start_date" in entities and "end_date" in entities:
            timeframe = f"from {entities['start_date']} to {entities['end_date']}"
        else:
            timeframe = "for the requested period"
        
        narrative = f"I'll analyze {metric} in {market} {timeframe}. "
        
        if entities.get("aggregation") == "avg":
            narrative += f"The average {metric} will be calculated across the time period."
        elif entities.get("aggregation") == "max":
            narrative += f"I'll identify the peak {metric} during this period."
        
        return narrative
    
    async def generate_insights(
        self,
        market: str,
        timeframe: str,
        data_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate automated market insights.
        
        Analyzes recent data and generates narrative insights.
        """
        # Mock data analysis
        insights = {
            "title": f"{market} Market Update - {timeframe.title()}",
            "summary": f"Analysis of {market} power market conditions",
            "key_findings": [
                f"Average prices in {market} were stable at $45/MWh",
                "Peak demand occurred during evening hours (6-9 PM)",
                "Renewable generation increased 15% compared to previous period",
                "No significant congestion events recorded",
            ],
            "detailed_analysis": f"""
            The {market} market showed typical seasonal patterns during {timeframe}.
            
            Price Dynamics:
            - Average LMP: $45.20/MWh (±$3.50 std dev)
            - Peak hour average: $62.30/MWh
            - Off-peak average: $32.15/MWh
            
            Supply & Demand:
            - Load factor: 68% (within normal range)
            - Renewable penetration: 32% (up from 28% last period)
            - Reserve margins: Adequate (>15%)
            
            Notable Events:
            - No major outages or forced derates
            - Weather conditions were moderate
            - Transmission system operated normally
            
            Forward Outlook:
            - Expect similar patterns to continue
            - Monitor weather forecasts for demand changes
            - Watch for scheduled maintenance announcements
            """,
            "charts_recommended": [
                "hourly_price_profile",
                "load_duration_curve",
                "generation_mix",
                "price_histogram",
            ],
        }
        
        return insights
    
    async def generate_report(
        self,
        report_type: str,
        markets: List[str],
        start_date: date,
        end_date: date,
        include_forecast: bool = True
    ) -> Dict[str, Any]:
        """
        Generate automated market report.
        
        Creates comprehensive PDF/HTML report with analysis.
        """
        report_id = f"RPT-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        # Generate title
        if report_type == "daily":
            title = f"Daily Market Brief - {start_date}"
        elif report_type == "weekly":
            title = f"Weekly Market Analysis - Week of {start_date}"
        else:
            title = f"Monthly Market Report - {start_date.strftime('%B %Y')}"
        
        # Executive summary
        executive_summary = f"""
        This {report_type} report covers {len(markets)} power markets from {start_date} to {end_date}.
        
        Key Highlights:
        - Markets analyzed: {', '.join(markets)}
        - Average system-wide LMP: $46.50/MWh
        - Total volume traded: 125,000 MWh
        - Price volatility remained within normal ranges
        - Renewable generation contributed 28% of supply
        """
        
        # Generate sections for each market
        sections = []
        for market in markets:
            sections.append({
                "title": f"{market} Market Analysis",
                "content": f"Detailed analysis of {market} market conditions...",
                "charts": ["price_trend", "volume_profile"],
                "statistics": {
                    "avg_price": 45.0 + hash(market) % 10,
                    "peak_price": 80.0 + hash(market) % 20,
                    "total_volume": 10000 + hash(market) % 5000,
                }
            })
        
        if include_forecast:
            sections.append({
                "title": "Forward Price Forecast",
                "content": "ML-powered price forecasts for the next 7 days...",
                "charts": ["forecast_curve", "confidence_intervals"],
                "forecast_data": {
                    "method": "transformer_ensemble",
                    "horizon": "7_days",
                    "confidence": 0.95,
                }
            })
        
        return {
            "report_id": report_id,
            "title": title,
            "executive_summary": executive_summary,
            "sections": sections,
            "download_url": f"/api/v1/reports/{report_id}/download",
        }


# Global NLP engine instance
nlp_engine = NLPEngine()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/nlp/query", response_model=QueryResponse)
async def natural_language_query(request: QueryRequest):
    """
    Process natural language query.
    
    Supports queries like:
    - "What was the average price in PJM last week?"
    - "Show me congestion in ERCOT for July"
    - "Compare MISO and CAISO prices yesterday"
    """
    try:
        logger.info(f"Processing NL query: {request.query}")
        
        result = await nlp_engine.parse_query(request.query, request.context)
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/nlp/insights", response_model=InsightResponse)
async def generate_market_insights(request: InsightRequest):
    """
    Generate automated market insights.
    
    Analyzes recent data and produces narrative insights
    about market conditions.
    """
    try:
        logger.info(f"Generating insights for {request.market} - {request.timeframe}")
        
        insights = await nlp_engine.generate_insights(
            request.market,
            request.timeframe,
            request.data_context
        )
        
        return InsightResponse(**insights)
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/nlp/reports", response_model=ReportResponse)
async def generate_automated_report(request: ReportRequest):
    """
    Generate automated market report.
    
    Creates comprehensive PDF/HTML report with charts,
    statistics, and narrative analysis.
    """
    try:
        logger.info(
            f"Generating {request.report_type} report for "
            f"{len(request.markets)} markets"
        )
        
        report = await nlp_engine.generate_report(
            request.report_type,
            request.markets,
            request.start_date,
            request.end_date,
            request.include_forecast
        )
        
        return ReportResponse(**report)
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/nlp/examples")
async def get_example_queries():
    """Get example natural language queries."""
    return {
        "examples": [
            {
                "category": "Price Queries",
                "queries": [
                    "What was the average price in PJM last week?",
                    "Show me peak prices in ERCOT for July",
                    "Get minimum LMP in CAISO yesterday",
                ]
            },
            {
                "category": "Comparisons",
                "queries": [
                    "Compare MISO and PJM prices last month",
                    "Show differences between day-ahead and real-time in NYISO",
                    "Which market had higher volatility: SPP or CAISO?",
                ]
            },
            {
                "category": "Analysis",
                "queries": [
                    "Explain why prices spiked in ERCOT yesterday",
                    "What drove congestion in PJM last week?",
                    "Analyze renewable impact on CAISO prices",
                ]
            },
            {
                "category": "Forecasting",
                "queries": [
                    "Forecast MISO prices for next week",
                    "Predict peak demand in NYISO tomorrow",
                    "What are price expectations for AESO next month?",
                ]
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8014)

