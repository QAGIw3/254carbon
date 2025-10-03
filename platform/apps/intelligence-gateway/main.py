"""
Intelligence Gateway

Unified API for all AI features with knowledge graph integration.
Routes queries to appropriate services and aggregates responses.
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligence Gateway",
    description="Unified API for AI-powered market intelligence",
    version="1.0.0",
)


class QueryType(str, Enum):
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"


class IntelligenceRequest(BaseModel):
    """Unified intelligence query."""
    query: str
    query_type: Optional[QueryType] = None
    markets: List[str] = []
    include_forecast: bool = False
    include_risk: bool = False
    language: str = "en"


class IntelligenceResponse(BaseModel):
    """Unified intelligence response."""
    query_id: str
    query_type: QueryType
    response: str
    data_insights: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    sources: List[str]
    related_queries: List[str]
    confidence: float


class KnowledgeGraphNode(BaseModel):
    """Knowledge graph node."""
    node_id: str
    node_type: str  # market, fuel, policy, company, asset
    name: str
    properties: Dict[str, Any]


class KnowledgeGraphRelationship(BaseModel):
    """Knowledge graph relationship."""
    from_node: str
    to_node: str
    relationship_type: str  # connects_to, depends_on, competes_with, etc.
    strength: float  # 0-1
    properties: Dict[str, Any]


class IntelligenceEngine:
    """Routes queries to appropriate AI services."""
    
    def __init__(self):
        self.services = {
            "copilot": "http://ai-copilot:8017",
            "nlp": "http://nlp-service:8014",
            "ml": "http://ml-service:8006",
            "risk": "http://risk-service:8008",
            "signals": "http://signals-service:8016",
        }
    
    async def process_query(
        self,
        query: str,
        query_type: Optional[QueryType],
        markets: List[str],
        include_forecast: bool,
        include_risk: bool,
        language: str
    ) -> Dict[str, Any]:
        """Process unified intelligence query."""
        
        # Auto-detect query type if not specified
        if not query_type:
            query_type = self._detect_query_type(query)
        
        # Route to appropriate services
        responses = {}
        
        # Always use copilot for conversational responses
        if query_type == QueryType.CONVERSATIONAL:
            responses["copilot"] = await self._call_copilot(query, language)
        
        # Add analytical insights
        if query_type in [QueryType.ANALYTICAL, QueryType.COMPARATIVE]:
            responses["nlp"] = await self._call_nlp(query, markets)
        
        # Add forecasts if requested
        if include_forecast or query_type == QueryType.PREDICTIVE:
            responses["forecast"] = await self._call_ml_forecast(markets)
        
        # Add risk analysis if requested
        if include_risk:
            responses["risk"] = await self._call_risk_analysis(markets)
        
        # Aggregate responses
        aggregated = self._aggregate_responses(responses)
        
        return aggregated
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Auto-detect query type from text."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["forecast", "predict", "future", "outlook"]):
            return QueryType.PREDICTIVE
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return QueryType.COMPARATIVE
        elif any(word in query_lower for word in ["analyze", "why", "explain", "how"]):
            return QueryType.ANALYTICAL
        else:
            return QueryType.CONVERSATIONAL
    
    async def _call_copilot(self, query: str, language: str) -> Dict:
        """Call AI Copilot service."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.services['copilot']}/api/v1/copilot/chat",
                    json={"query": query, "language": language},
                    timeout=30.0
                )
                return response.json()
            except Exception as e:
                logger.error(f"Error calling copilot: {e}")
                return {"response": "Service temporarily unavailable"}
    
    async def _call_nlp(self, query: str, markets: List[str]) -> Dict:
        """Call NLP service for insights."""
        # Mock call
        return {
            "insights": "Market analysis results...",
            "key_findings": ["Finding 1", "Finding 2"],
        }
    
    async def _call_ml_forecast(self, markets: List[str]) -> Dict:
        """Call ML service for forecasts."""
        # Mock call
        return {
            "forecasts": {market: {"1h": 45.0, "1d": 46.5} for market in markets}
        }
    
    async def _call_risk_analysis(self, markets: List[str]) -> Dict:
        """Call risk service."""
        # Mock call
        return {
            "var_95": 1500.0,
            "var_99": 2200.0,
        }
    
    def _aggregate_responses(self, responses: Dict) -> Dict[str, Any]:
        """Aggregate responses from multiple services."""
        aggregated = {
            "response": "",
            "data_insights": {},
            "visualizations": [],
            "sources": [],
        }
        
        if "copilot" in responses:
            aggregated["response"] = responses["copilot"].get("response", "")
            aggregated["sources"].extend(responses["copilot"].get("sources", []))
        
        if "nlp" in responses:
            aggregated["data_insights"]["analysis"] = responses["nlp"]
        
        if "forecast" in responses:
            aggregated["data_insights"]["forecasts"] = responses["forecast"]
        
        if "risk" in responses:
            aggregated["data_insights"]["risk"] = responses["risk"]
        
        return aggregated


class KnowledgeGraph:
    """Knowledge graph for market relationships."""
    
    def __init__(self):
        self.nodes = []
        self.relationships = []
        self._build_initial_graph()
    
    def _build_initial_graph(self):
        """Build initial knowledge graph."""
        # Create market nodes
        markets = ["PJM", "MISO", "CAISO", "ERCOT", "NYISO", "SPP"]
        for market in markets:
            self.nodes.append({
                "node_id": f"market_{market.lower()}",
                "node_type": "market",
                "name": market,
                "properties": {"region": "North America"},
            })
        
        # Create fuel nodes
        fuels = ["Natural Gas", "Coal", "Nuclear", "Wind", "Solar", "Hydro"]
        for fuel in fuels:
            self.nodes.append({
                "node_id": f"fuel_{fuel.lower().replace(' ', '_')}",
                "node_type": "fuel",
                "name": fuel,
                "properties": {"category": "renewable" if fuel in ["Wind", "Solar", "Hydro"] else "conventional"},
            })
        
        # Create relationships
        # Example: PJM depends on natural gas
        self.relationships.append({
            "from_node": "market_pjm",
            "to_node": "fuel_natural_gas",
            "relationship_type": "depends_on",
            "strength": 0.45,  # 45% gas generation
            "properties": {"generation_share": 0.45},
        })
        
        # PJM connects to MISO
        self.relationships.append({
            "from_node": "market_pjm",
            "to_node": "market_miso",
            "relationship_type": "connects_to",
            "strength": 0.80,
            "properties": {"transmission_capacity_mw": 5000},
        })
    
    def query(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Query knowledge graph."""
        # Find node
        node = next((n for n in self.nodes if n["node_id"] == node_id), None)
        if not node:
            return {}
        
        # Find related nodes
        related = []
        for rel in self.relationships:
            if rel["from_node"] == node_id:
                related_node = next((n for n in self.nodes if n["node_id"] == rel["to_node"]), None)
                if related_node:
                    related.append({
                        "node": related_node,
                        "relationship": rel["relationship_type"],
                        "strength": rel["strength"],
                    })
        
        return {
            "node": node,
            "related_nodes": related,
        }


# Global instances
engine = IntelligenceEngine()
knowledge_graph = KnowledgeGraph()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "intelligence-gateway"}


@app.post("/api/v1/intelligence/query", response_model=IntelligenceResponse)
async def unified_query(request: IntelligenceRequest):
    """
    Unified intelligence query endpoint.
    
    Routes to appropriate services and aggregates responses.
    """
    try:
        logger.info(f"Processing intelligence query: {request.query[:50]}...")
        
        result = await engine.process_query(
            request.query,
            request.query_type,
            request.markets,
            request.include_forecast,
            request.include_risk,
            request.language
        )
        
        query_id = f"Q-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        return IntelligenceResponse(
            query_id=query_id,
            query_type=request.query_type or QueryType.CONVERSATIONAL,
            response=result.get("response", ""),
            data_insights=result.get("data_insights", {}),
            visualizations=result.get("visualizations", []),
            sources=result.get("sources", []),
            related_queries=["What about CAISO?", "Show me the forecast"],
            confidence=0.90,
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/intelligence/knowledge-graph/node/{node_id}")
async def get_knowledge_graph_node(node_id: str):
    """
    Get knowledge graph node and relationships.
    
    Explore connections between markets, fuels, policies, etc.
    """
    result = knowledge_graph.query(node_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return result


@app.get("/api/v1/intelligence/knowledge-graph/search")
async def search_knowledge_graph(
    query: str,
    node_type: Optional[str] = None,
):
    """Search knowledge graph."""
    # Filter nodes by query
    matching_nodes = [
        n for n in knowledge_graph.nodes
        if query.lower() in n["name"].lower()
        and (not node_type or n["node_type"] == node_type)
    ]
    
    return {"nodes": matching_nodes[:20]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)

