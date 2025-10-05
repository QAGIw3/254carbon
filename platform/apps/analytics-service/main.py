"""
Analytics Service

Unified ML, risk, and market analytics service consolidating:
- ML forecasting (from ml-service)
- Risk analytics (from risk-service)  
- LMP decomposition (from lmp-decomposition-service)
- Market insights (from market-insights)
- Satellite intelligence (from satellite-intel)
- Quantum optimization (from quantum-optimizer)
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# MLflow integration
import mlflow

# Import shared components
from forecasting.models import ModelRegistry
from forecasting.feature_engineering import FeatureEngineer
from forecasting.training import ModelTrainer
from forecasting.retraining_pipeline import RetrainingPipeline

# Import routers
from routers import research_api
from routers import refining_api
from routers import renewables_api
from routers import supply_chain_api
from routers import portfolio_api
from routers import arbitrage_api
from routers import transition_api
from routers import carbon_api
from routers import insights
from routers import satellite
from routers import quantum

# Import risk components
from risk.var_calculator import VaRCalculator
from risk.portfolio import PortfolioAggregator
from risk.stress_testing import StressTestEngine

# Import decomposition components (imported but used by decomposition router)
# from decomposition.decomposer import LMPDecomposer
# from decomposition.ptdf_calculator import PTDFCalculator
# from decomposition.basis_surface import BasisSurfaceModeler

# Import insights, satellite, quantum engines (already instantiated in routers)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Analytics Service",
    description="Unified ML, risk, and market analytics platform",
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

# Initialize MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# Initialize forecasting components (shared across service)
model_registry = ModelRegistry()
feature_engineer = FeatureEngineer()
trainer = ModelTrainer()
retraining_pipeline = RetrainingPipeline()

# Initialize risk components (shared across service)
var_calculator = VaRCalculator()
portfolio_aggregator = PortfolioAggregator()
stress_engine = StressTestEngine()

logger.info("Analytics Service components initialized")

# Include all routers
# ML/Forecasting routers (from ml-service)
app.include_router(research_api.router)
app.include_router(refining_api.router)
app.include_router(renewables_api.router)
app.include_router(supply_chain_api.router)
app.include_router(portfolio_api.router)
app.include_router(arbitrage_api.router)
app.include_router(transition_api.router)
app.include_router(carbon_api.router)

# New unified routers
app.include_router(insights.router)
app.include_router(satellite.router)
app.include_router(quantum.router)

# TODO: Add decomposition and risk routers
# app.include_router(decomposition_router)
# app.include_router(risk_router)

logger.info("Analytics Service routers registered")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "analytics-service",
        "version": "2.0.0",
        "models_loaded": model_registry.count() if hasattr(model_registry, 'count') else 0,
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Analytics Service",
        "version": "2.0.0",
        "description": "Unified analytics platform",
        "modules": [
            "forecasting",
            "risk",
            "decomposition",
            "insights",
            "satellite",
            "quantum",
        ],
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)

