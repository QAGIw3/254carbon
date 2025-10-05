# Analytics Service

**Version:** 2.0.0  
**Port:** 8008  
**Status:** Operational

## Overview

The Analytics Service is a consolidated platform that unifies all ML, risk, and market analytics capabilities previously scattered across 6 separate services. This consolidation reduces operational complexity, enables shared infrastructure, and improves resource utilization.

## Consolidated Services

This service consolidates the following previous services:

1. **ml-service** (Port 8006) → Forecasting Module
2. **risk-service** (Port 8008) → Risk Module
3. **lmp-decomposition-service** (Port 8009) → Decomposition Module
4. **market-insights** (Port 8015) → Insights Module
5. **satellite-intel** → Satellite Module
6. **quantum-optimizer** → Quantum Module

## Consolidation Roadmap

This service will be migrated into `services/analytics-platform` as part of the modular monolith refactor.

- Shared utilities now live under `libs/python/src/carbon254/analytics`.
- Public API contracts will move to `services/analytics-platform` with versioned OpenAPI specs.
- During migration, routes remain stable; plan for phased cutoff using feature flags.

## Architecture

```
analytics-service/
├── forecasting/              # ML forecasting & training
│   ├── models.py
│   ├── training.py
│   ├── feature_engineering.py
│   └── retraining_pipeline.py
├── risk/                     # Risk analytics
│   ├── var_calculator.py
│   ├── portfolio.py
│   └── stress_testing.py
├── decomposition/            # LMP decomposition
│   ├── decomposer.py
│   ├── ptdf_calculator.py
│   └── basis_surface.py
├── insights/                 # Market insights
│   └── engine.py
├── satellite/                # Satellite intelligence
│   └── intelligence.py
├── quantum/                  # Quantum optimization
│   └── optimizer.py
└── routers/                  # API routers
    ├── forecasting.py
    ├── risk.py
    ├── decomposition.py
    ├── insights.py
    ├── satellite.py
    ├── quantum.py
    ├── research.py
    ├── refining.py
    ├── renewables.py
    ├── supply_chain.py
    ├── portfolio.py
    ├── arbitrage.py
    ├── transition.py
    └── carbon.py
```

## Key Features

### Forecasting
- ML-based price forecasting (XGBoost, LightGBM, Transformers)
- Multi-commodity multimodal transformer models
- Feature engineering and model training
- MLflow integration for experiment tracking
- Automated retraining pipelines

### Risk Analytics
- Value at Risk (VaR) calculations
- Expected Shortfall (CVaR)
- Portfolio aggregation
- Stress testing
- Historical, parametric, and Monte Carlo methods

### LMP Decomposition
- Energy component analysis
- Congestion component calculation
- Loss component estimation
- PTDF calculations
- Basis surface modeling

### Market Insights
- Real-time anomaly detection
- Arbitrage opportunity identification
- Fundamental driver analysis
- Daily market briefings

### Satellite Intelligence
- Oil storage tank level monitoring
- Coal stockpile volume estimation
- Solar/wind farm operational status
- Pipeline integrity monitoring

### Quantum Optimization
- Portfolio optimization (quantum algorithms)
- Unit commitment optimization
- Transmission flow optimization
- Risk scenario generation

## API Endpoints

### Forecasting
- `POST /api/v1/ml/forecast` - Generate price forecast
- `POST /api/v1/ml/train` - Train new model
- `POST /api/v1/ml/retrain` - Retrain existing model

### Risk
- `POST /api/v1/risk/var` - Calculate VaR
- `POST /api/v1/risk/metrics` - Portfolio risk metrics
- `POST /api/v1/risk/stress` - Stress testing

### Insights
- `GET /api/v1/insights/anomalies` - Detect anomalies
- `GET /api/v1/insights/arbitrage` - Find arbitrage opportunities
- `GET /api/v1/insights/daily-briefing` - Daily market briefing

### Satellite
- `GET /api/v1/satellite/oil-storage/{tank_id}` - Oil tank level
- `GET /api/v1/satellite/coal-stockpile/{site_id}` - Coal stockpile
- `GET /api/v1/satellite/solar-farm/{farm_id}` - Solar farm status

### Quantum
- `POST /api/v1/quantum/portfolio` - Portfolio optimization
- `POST /api/v1/quantum/unit-commitment` - Unit commitment
- `POST /api/v1/quantum/transmission` - Transmission flow

## Shared Infrastructure

- **MLflow**: Unified experiment tracking and model registry
- **Model Registry**: Shared model storage (reduces memory 30%)
- **GPU Allocation**: Single unified GPU resource pool
- **Database Connections**: Shared ClickHouse, PostgreSQL, Redis connections

## Environment Variables

```bash
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=9000
REDIS_HOST=redis
REDIS_PORT=6379
POSTGRES_HOST=postgresql
POSTGRES_PORT=5432
MLFLOW_TRACKING_URI=http://mlflow:5000
```

## Running the Service

### Local Development
```bash
cd platform/apps/analytics-service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8008
```

### Docker
```bash
docker build -t 254carbon/analytics-service:latest .
docker run -p 8008:8008 254carbon/analytics-service:latest
```

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

## Benefits of Consolidation

1. **Operational**: 6 services → 1 service (83% reduction)
2. **Performance**: Shared ML model cache, reduced memory usage
3. **Development**: Single codebase, easier dependency management
4. **Cost**: Reduced infrastructure overhead, shared GPU utilization

## Migration Notes

The service maintains API compatibility with previous services. All existing endpoints remain accessible with the same request/response formats.

## Health Check

```bash
curl http://analytics-service:8008/health
```

## Documentation

- API Docs: http://analytics-service:8008/docs
- OpenAPI: http://analytics-service:8008/openapi.json

