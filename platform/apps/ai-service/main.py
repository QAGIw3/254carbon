"""
AI Service

Unified LLM-based services consolidating:
- AI Copilot (from ai-copilot)
- NLP Query Understanding (from nlp-service)
- RegTech Compliance AI (from regtech-ai)
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
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

