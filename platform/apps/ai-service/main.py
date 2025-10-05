"""
AI Service

Overview
--------
Unified LLM-based services consolidating multiple AI capabilities into a single
FastAPI application:

- AI Copilot (from ai-copilot) for conversational assistance and RAG
- NLP Query Understanding (from nlp-service) for intent/entity parsing
- RegTech Compliance AI (from regtech-ai) for regulatory analytics

Key Endpoints
-------------
- `/api/v1/copilot/*`: chat (REST) and `/ws/{conversation_id}` (WebSocket)
- `/api/v1/nlp/*`: natural language query parsing and insights
- `/api/v1/regtech/*`: regulatory updates, gap analysis, reports
- `/health`: liveness/readiness probe
- `/docs`: OpenAPI UI

Environment
-----------
External keys such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are optional in
development but should be configured in production via Kubernetes secrets.

Usage
-----
Run locally with `uvicorn main:app --reload --port 8020`.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from routers import copilot
from routers import nlp
from routers import regtech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Service",
    description="Unified NLP, AI Copilot, and RegTech AI platform",
    version="2.0.0",
)

# CORS middleware
# Allow cross-origin requests for browser-based tooling and integrations.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
# Each router groups related endpoints and delegates to engines.
app.include_router(copilot.router)
app.include_router(nlp.router)
app.include_router(regtech.router)

logger.info("AI Service initialized with all routers")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-service",
        "version": "2.0.0",
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AI Service",
        "version": "2.0.0",
        "description": "Unified LLM-based AI platform",
        "modules": [
            "copilot",
            "nlp",
            "regtech",
        ],
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
