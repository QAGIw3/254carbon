"""
NLP Service API Router

Purpose
-------
Transforms natural language into structured actions and generates light-weight
insights. Delegates to `NLPEngine` for intent classification and entity
extraction, and optionally emits SQL/API call plans and narrative text.

Endpoints
---------
- POST `/api/v1/nlp/query` — parse NL query into structure
- POST `/api/v1/nlp/insights` — generate short market insights
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from nlp.query_parser import NLPEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/nlp", tags=["nlp"])

# Initialize NLP engine
nlp_engine = NLPEngine()


class QueryRequest(BaseModel):
    """Natural language query."""
    query: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Parsed query response."""
    intent: str
    entities: Dict[str, Any]
    sql_query: Optional[str]
    api_calls: Optional[List[Dict]]
    narrative_response: str


class InsightRequest(BaseModel):
    """Market insight generation request."""
    market: str
    timeframe: str
    data_context: Optional[Dict] = None


class InsightResponse(BaseModel):
    """Market insight response."""
    title: str
    summary: str
    key_findings: List[str]
    detailed_analysis: str


@router.post("/query", response_model=QueryResponse)
async def natural_language_query(request: QueryRequest):
    """Process natural language query."""
    try:
        logger.info(f"Processing NL query: {request.query}")
        
        result = await nlp_engine.parse_query(request.query, request.context)
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insights", response_model=InsightResponse)
async def generate_market_insights(request: InsightRequest):
    """Generate automated market insights."""
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
