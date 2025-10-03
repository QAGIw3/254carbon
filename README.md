# 254Carbon - AI Market Intelligence Platform

**See the market. Price the future.**

A comprehensive energy and commodity market intelligence platform delivering real-time prices, forward curves, and scenario forecasts to 2050.

## Overview

The 254Carbon platform provides:
- **Real-time data streaming**: Vendor ticks to client ≤2s latency
- **Forward curves**: Monthly forecasts to 2050 with scenario modeling
- **Multi-market coverage**: US+Canada power (ISO/RTO nodal), gas hubs, REC, LCFS, LNG basis
- **Flexible delivery**: Web hub, REST APIs, and signed downloads
- **Scenario engine**: User-adjustable assumptions with full reproducibility

## Architecture

- **Infrastructure**: Local Kubernetes cluster
- **Data Layer**: Kafka (streaming), ClickHouse (OLAP), PostgreSQL (metadata), MinIO (object storage)
- **Services**: API Gateway, Curve Service, Scenario Engine, Report Service, Download Center
- **Frontend**: React TypeScript SPA
- **Security**: Keycloak OIDC, TLS everywhere, audit logging
- **Observability**: Prometheus + Grafana

## Project Structure

```
/platform
  /apps              # Application services
    /gateway         # FastAPI API Gateway with OIDC
    /web-hub         # React TypeScript SPA
    /curve-service   # Forward curve generation & QP solver
    /scenario-engine # Scenario DSL parser and execution
    /report-service  # HTML/PDF report generation
    /download-center # Export and signed URL service
  /data              # Data ingestion and processing
    /ingestion-orch  # Apache Airflow orchestration
    /connectors      # Source plugins (MISO, CAISO, etc.)
    /schemas         # Avro/JSON schemas
  /infra             # Infrastructure as Code
    /helm            # Helm charts
    /k8s             # Kubernetes manifests
    /observability   # Prometheus/Grafana config
  /docs              # Documentation
  /ci                # CI/CD pipelines
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Local Kubernetes (kind or k3d)
- kubectl and helm CLI tools
- Python 3.11+
- Node.js 18+

### Local Development

1. **Start infrastructure services:**
   ```bash
   cd infra/helm
   ./deploy-infrastructure.sh
   ```

2. **Initialize databases:**
   ```bash
   cd data/schemas
   ./init-databases.sh
   ```

3. **Start backend services:**
   ```bash
   cd apps
   ./start-services.sh
   ```

4. **Start frontend:**
   ```bash
   cd apps/web-hub
   npm install
   npm run dev
   ```

5. **Access the platform:**
   - Web Hub: http://localhost:3000
   - API Gateway: http://localhost:8000
   - Grafana: http://localhost:3001

## Pilot Markets

- **MISO**: Full access (Hub + API + Downloads)
- **CAISO**: Hub and Downloads only (API disabled in pilot)

## Key Features

### Data Products

- **Live Prices**: RT/DA nodal with ≤5 min freshness
- **Forward Curves**: Monthly to 2050, quarterly to 10Y, annual beyond
- **Fundamentals**: Load, generation, capacity, storage, policy
- **Environmental**: REC and LCFS compliance tracking

### Scenario Modeling

- User-adjustable macro, fuel, power, and policy assumptions
- Reproducible runs with full lineage tracking
- Saved scenarios and versioning
- Custom new builds and retirements

### Quality Gates

- Stream latency p95 ≤5s
- Data completeness ≥99.5%
- Forecast MAPE ≤12% (months 1-6)
- 99.9% uptime SLA

## Development

### Running Tests

```bash
# Unit tests
pytest apps/*/tests/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/
```

### Code Style

```bash
# Format code
black .
isort .

# Type checking
mypy apps/

# Linting
ruff check .
```

## Deployment

The platform uses GitLab CI/CD with automated testing and Helm-based deployment:

1. Lint and type checking
2. Unit tests
3. Integration tests (with test infrastructure)
4. Image build and scan
5. Helm package and deploy
6. Smoke tests

## Security

- TLS encryption everywhere
- At-rest encryption for all data stores
- OIDC authentication via Keycloak
- Role-based access control
- Audit logging for all data access
- Network policies and pod security standards
- Regular secrets rotation

## Monitoring

- Prometheus metrics for all services
- Grafana dashboards per market
- Data quality alerts
- SLA monitoring
- Distributed tracing

## License

Proprietary - 254Carbon, Inc.

## Support

For issues and support: support@254carbon.ai
