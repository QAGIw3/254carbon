# 254Carbon Market Intelligence Platform

**Enterprise-grade market data platform for power, gas, and carbon markets**

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![SOC2](https://img.shields.io/badge/SOC2-Compliant-green.svg)]()
[![Uptime](https://img.shields.io/badge/uptime-99.9%25-brightgreen.svg)]()

---

## 🎯 Overview

The 254Carbon Market Intelligence Platform is a comprehensive solution for real-time market data ingestion, forward curve generation, scenario modeling, and forecast backtesting across global energy markets.

### Key Features

- **Real-Time Data Ingestion**: Sub-2-second latency streaming from 26+ power markets
- **Forward Curve Generation**: QP-optimized curves with scenario-based forecasting
- **Backtesting Pipeline**: MAPE/WAPE/RMSE validation with automated accuracy monitoring
- **Entitlements System**: Multi-tenant architecture with granular access controls
- **Production-Ready**: SOC2 compliant with comprehensive security and monitoring

### Markets Supported

- **North America**: MISO, CAISO, PJM, ERCOT, SPP, NYISO, IESO, AESO
- **Europe**: EPEX, Nord Pool, Poland (TGE), Eastern Europe markets
- **APAC**: JEPX (Japan), NEM (Australia), Korea, Singapore
- **Emerging**: China, India, Brazil, Mexico, Nigeria

---

## 🏗️ Architecture

### Modular Monolith Consolidation

The platform is actively migrating from dozens of microservices to domain-aligned modular monoliths. The new layout under `services/` consolidates functionality without sacrificing clear boundaries:

```text
services/
  edge-gateway/        # Unified API & GraphQL edge layer
  analytics-platform/  # Forecasting, risk, research, ML APIs
  data-platform/       # Connectors, streaming, data quality
  identity-platform/   # AuthN/Z, entitlements, audit
  observability/       # Metrics, logging, tracing
libs/
  python/              # Shared Python packages (config, db, http, analytics)
  js/                  # Shared TypeScript clients and utilities
contracts/
  http/ graphql/ events/  # Source of truth for external and event contracts
```

- Shared utilities are exported via `carbon254-*` packages for Python/TypeScript consumers.
- Contracts drive client generation and compatibility tests; see `contracts/http/edge-gateway/openapi.yaml`.
- Existing microservices remain under `platform/apps/` until migration completes; see `docs/architecture/service-inventory.md` for the mapping.

```
External APIs → Connectors → Kafka → ClickHouse
                                  ↓
                            Curve Service → Forward Curves
                                  ↓
                            Scenario Engine → Forecasts
                                  ↓
                        API Gateway ← Web Hub (Users)
```

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React, TypeScript, Tailwind CSS, Vite |
| **API Layer** | FastAPI, Python 3.11, WebSockets |
| **Data Stores** | ClickHouse (OLAP), PostgreSQL (metadata) |
| **Streaming** | Apache Kafka, Avro schemas |
| **Orchestration** | Apache Airflow, Kubernetes, Helm |
| **Authentication** | Keycloak (OIDC), OAuth 2.0 |
| **Monitoring** | Prometheus, Grafana, structured logging |
| **Object Storage** | MinIO (S3-compatible) |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Kubernetes cluster (for production)
- Python 3.11+
- Node.js 18+

### Local Development

```bash
# Clone repository
git clone https://github.com/254carbon/market-intelligence.git
cd market-intelligence

# Start infrastructure services
cd platform
docker-compose up -d

# Initialize databases
./infra/scripts/init-databases.sh

# Start services
make dev

# Access points
# Web Hub: http://localhost:3000
# API Gateway: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Grafana: http://localhost:3001
```

### Production Deployment

```bash
# Deploy infrastructure (PostgreSQL, ClickHouse, Kafka, etc.)
make infra

# Initialize databases
make init-db

# Deploy application services
helm upgrade --install market-intelligence \
  platform/infra/helm/market-intelligence \
  --namespace market-intelligence \
  --create-namespace

# Verify deployment
kubectl get pods -n market-intelligence
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for comprehensive deployment guide.

---

## 🌐 Ingress & Endpoints (Kubernetes)

Ingress host: `254carbon.local`

- Web Hub: `http://254carbon.local/web` (also `http://254carbon.local/`)
- API Gateway: `http://254carbon.local/api` (OpenAPI docs at `/api/docs`)

### Commodities Service Consolidation

We consolidated multiple commodity-specific services into a single
Commodities Service running on port `8012` with a unified surface at
`/api/v1/commodities/*`.

- New service: `platform/apps/commodities-service` (FastAPI)
- Gateway proxy: forwards `/api/v1/commodities/*` to the commodities service
- Ingress: routes the same prefix directly to the commodities service to reduce hops
- Old singletons: `gas-service` and `battery-materials` were removed after migration
- Back-compat: legacy gateway commodity endpoints remain, now carrying
  `Deprecation`, `Sunset`, and `Link` headers to guide clients to the new paths

Example new routes:

- `GET /api/v1/commodities/gas/prices`
- `GET /api/v1/commodities/oil/curves`
- `GET /api/v1/commodities/coal/indices`
- `GET /api/v1/commodities/biofuels/rin-prices`
- `GET /api/v1/commodities/battery-materials/lithium`

See `platform/apps/commodities-service/README.md` for details on env vars,
caching, fallbacks, and testing.

Configure local DNS for the Ingress IP:

```
# Get the Ingress Controller load balancer IP (MetalLB)
kubectl get svc -n ingress-nginx ingress-nginx-controller \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}{"\n"}'

# Add to /etc/hosts (replace ${IP})
echo ${IP} 254carbon.local | sudo tee -a /etc/hosts
```

Notes for internal services (not exposed via Ingress):

- MinIO Console/API (dev): `kubectl -n market-intelligence port-forward svc/minio 9000:9000 9001:9001` → `http://localhost:9001`
- Keycloak (dev): `kubectl -n market-intelligence port-forward deploy/keycloak 8080:8080` → `http://localhost:8080`
- ClickHouse HTTP (dev): `kubectl -n market-intelligence port-forward svc/clickhouse 8123:8123` → `http://localhost:8123/ping`

Service summary via Ingress:

- `web-hub` → `/web` and `/`
- `gateway` → `/api`

To add more services to Ingress, edit `ingress.yaml` and add additional `paths` entries.

---

## 📊 Core Services

### Edge Gateway
- **Purpose**: Consolidated edge layer for REST and GraphQL entrypoints
- **Location**: `/services/edge-gateway`
- **Contracts**: `contracts/http/edge-gateway/openapi.yaml`, `contracts/graphql/edge-gateway`
- **Features**: OIDC auth, rate limit middleware, circuit breakers, contract-first APIs

### Analytics Platform
- **Purpose**: Unified analytics, forecasting, and research APIs
- **Location**: `/services/analytics-platform`
- **Features**: Shared feature engineering (`carbon254.analytics`), consolidated routers, scenario tooling
- **Features**: Smooth curves, tenor reconciliation, scenario support, lineage tracking

### Data Platform
- **Purpose**: Ingestion, streaming, and data quality operations
- **Location**: `/services/data-platform`
- **Features**: Kafka producers/consumers (`carbon254.kafka`), schema validation, connector orchestration
- **Features**: DSL parser, assumption management, run tracking

### Identity Platform
- **Purpose**: Authentication, authorization, and entitlements
- **Location**: `/services/identity-platform`
- **Features**: Unified token services, entitlements engine, audit logging

### Download Center
- **Purpose**: Data export and bulk download management
- **Location**: `/platform/apps/download-center/`
- **Features**: CSV/Parquet export, signed URLs, entitlement enforcement

### Observability Platform
- **Purpose**: Metrics, logging, tracing foundation
- **Location**: `/services/observability`
- **Features**: Prometheus exporters, OpenTelemetry configuration, shared logging helpers

### Routing Service
- **Purpose**: Multi-source redundancy with intelligent failover, blending, and trust scoring
- **Location**: `/platform/apps/routing-service/`
- **Features**: Automatic source selection, circuit breaker, synthetic fallback, full audit trail
- **Documentation**: [Design Document](docs/ROUTING_SERVICE_DESIGN.md) | [Integration Guide](platform/apps/routing-service/INTEGRATION.md)

---

## 🔌 Data Connectors

### Connector SDK

Base class for all data source integrations with standard lifecycle:

```python
from platform.data.connectors.base import Ingestor

class MyConnector(Ingestor):
    def discover(self) -> Dict[str, Any]: ...
    def pull_or_subscribe(self) -> Iterator[Dict]: ...
    def map_to_schema(self, raw: Dict) -> Dict: ...
    def emit(self, events: Iterator[Dict]) -> int: ...
    def checkpoint(self, state: Dict) -> None: ...
```

### Implemented Connectors

**North America:**
- **MISO**: Real-time and day-ahead LMP data
- **CAISO**: Hub-only data with entitlement restrictions (pilot)
- **PJM**, **ERCOT**, **NYISO**: US ISOs
- **AESO**: Alberta pool price and demand ✨ NEW
- **IESO**: Ontario HOEP and generation mix ✨ NEW

**Europe:**
- **EPEX**, **Nord Pool**: European power exchanges
- **ENTSOE**: Pan-European transparency platform

**APAC:**
- **NEM**: Australian 5-region spot market ✨ NEW
- **JEPX**: Japan Electric Power Exchange

**Latin America:**
- **Brazil ONS**: PLD prices and hydro reservoirs ✨ NEW

**Infrastructure & Storage:** ✨ NEW
- **ALSI LNG**: European LNG terminal inventory and flows (GIE)
- **REexplorer**: Renewable energy resource assessments (NREL)
- **WRI Power Plants**: Global power plant database (~30k plants)
- **GEM Transmission**: Transmission infrastructure and projects

See `/platform/data/connectors/` for complete list

### Orchestration

Airflow DAGs schedule connector runs with data quality checks:
- `/platform/data/ingestion-orch/dags/miso_ingestion_dag.py`
- `/platform/data/ingestion-orch/dags/caiso_ingestion_dag.py`
- `/platform/data/ingestion-orch/dags/aeso_ingestion_dag.py`
- `/platform/data/ingestion-orch/dags/ieso_ingestion_dag.py` ✨ NEW
- `/platform/data/ingestion-orch/dags/nem_ingestion_dag.py` ✨ NEW
- `/platform/data/ingestion-orch/dags/brazil_ons_ingestion_dag.py` ✨ NEW
- `/platform/data/ingestion-orch/dags/agsi_ingestion_dag.py` ✨ NEW
- `/platform/data/ingestion-orch/dags/coal_stockpile_monitoring_dag.py` ✨ NEW

---

## 🔐 Security & Compliance

### SOC2 Compliance

The platform implements comprehensive security controls for SOC2 Type II compliance:

- ✅ **Authentication**: OIDC with MFA for admin accounts
- ✅ **Authorization**: RBAC and entitlements system
- ✅ **Encryption**: TLS 1.3 in transit, AES-256 at rest
- ✅ **Audit Logging**: Immutable audit trail with 2-year retention
- ✅ **Secrets Management**: Automated rotation with External Secrets Operator
- ✅ **Network Security**: Zero-trust with IP allowlisting
- ✅ **Monitoring**: Real-time security event detection and alerting

See [docs/SOC2_COMPLIANCE.md](./platform/docs/SOC2_COMPLIANCE.md) for detailed documentation.

### Security Features

| Feature | Implementation | Location |
|---------|---------------|----------|
| IP Allowlisting | Ingress-level restrictions | `/platform/infra/k8s/security/ip-allowlist.yaml` |
| Rate Limiting | 100 req/s general, 1 req/s exports | Nginx ingress annotations |
| Network Policies | Zero-trust, default deny | `/platform/infra/k8s/security/network-policies.yaml` |
| Secrets Rotation | Monthly automated rotation | `/platform/infra/scripts/rotate-secrets.sh` |
| Audit Logging | All API calls and data access | `/platform/shared/audit_logger.py` |

---

## 📈 Monitoring & SLAs

### Service Level Objectives

| Metric | Target | Current | Monitoring |
|--------|--------|---------|------------|
| **API Latency (p95)** | < 250ms | ✅ 180ms | Prometheus + Grafana |
| **Stream Latency** | < 2s | ✅ 1.2s | Custom metrics |
| **Data Completeness** | ≥ 99.5% | ✅ 99.8% | Data quality checks |
| **Uptime** | 99.9% | ✅ 99.95% | Health checks |
| **Error Rate** | < 1% | ✅ 0.2% | Error tracking |

### Grafana Dashboards

Four comprehensive dashboards for operational monitoring:

1. **Forecast Accuracy**: MAPE/WAPE/RMSE by market and horizon
2. **Data Quality**: Freshness, completeness, anomalies
3. **Service Health**: Latency, error rates, resource usage
4. **Security & Audit**: Failed auth, data exports, unusual activity

Location: `/platform/infra/monitoring/grafana/dashboards/`

### Alerting

- **Critical**: < 15 min response (data breach, system compromise)
- **High**: < 1 hour response (brute force, high latency)
- **Medium**: < 4 hours response (unusual exports, config changes)
- **Low**: < 24 hours response (minor violations)

---

## 🧪 Testing

### Load Testing

K6-based load tests validate SLA compliance:

```bash
cd platform/tests/load

# API load test (30 minutes)
./run-load-tests.sh

# Streaming test
k6 run streaming-load-test.js

# Results
open results/load-test-report.html
```

**Test Scenarios:**
- Ramp: 50 → 200 users over 15 minutes
- Sustained: 200 users for 10 minutes
- Spike: 500 users for 3 minutes
- Validates: p95 < 250ms, error rate < 1%

### Unit Tests

```bash
# Python services
pytest platform/apps/*/tests/

# Frontend
cd platform/apps/web-hub
npm test
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [DEPLOYMENT.md](./DEPLOYMENT.md) | Production deployment guide |
| [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) | Current implementation status |
| [SOC2_COMPLIANCE.md](./platform/docs/SOC2_COMPLIANCE.md) | Security and compliance documentation |
| [API Documentation](http://localhost:8000/docs) | Auto-generated OpenAPI docs |

---

## 🎓 Getting Started

### For Data Engineers

1. Review connector SDK: `/platform/data/connectors/base.py`
2. Examine example connectors: MISO, CAISO
3. Create new connector following pattern
4. Add Airflow DAG for scheduling
5. Configure data quality checks

### For Backend Developers

1. Explore API Gateway: `/platform/apps/gateway/main.py`
2. Review authentication flow: `/platform/apps/gateway/auth.py`
3. Understand entitlements: `/platform/apps/gateway/entitlements.py`
4. Add new endpoints following FastAPI patterns

### For Frontend Developers

1. Navigate to Web Hub: `/platform/apps/web-hub/`
2. Review component structure: `/src/components/`
3. Examine API integration: `/src/services/api.ts`
4. Study authentication: `/src/stores/authStore.ts`

---

## 🤝 Support & Contact

- **Technical Issues**: Open GitHub issue
- **Security Concerns**: security@254carbon.ai
- **Sales & Partnerships**: sales@254carbon.ai
- **Documentation**: https://docs.254carbon.ai

---

## 📜 License

Proprietary - © 2025 254Carbon. All rights reserved.

This software is confidential and proprietary to 254Carbon. Unauthorized copying, distribution, or use is strictly prohibited.

---

## ✨ Recent Updates

### October 2025 - Production MVP Complete

- ✅ Backtesting service with MAPE/WAPE/RMSE validation
- ✅ CAISO connector with pilot entitlement restrictions
- ✅ Comprehensive audit logging and security monitoring
- ✅ Automated secrets rotation with External Secrets Operator
- ✅ IP allowlisting and rate limiting at ingress
- ✅ 4 Grafana dashboards for operational monitoring
- ✅ K6 load tests validating SLA compliance
- ✅ SOC2 compliance documentation complete
- ✅ Airflow DAGs for MISO and CAISO orchestration

**Status**: Ready for production deployment with pilot customers

---

**Built with ❤️ by the 254Carbon team**

### Infrastructure Alignment
- Helmfile under `infra/helmfile` is the source of truth for deployments.
- Root-level Kubernetes manifests (`*-deployment.yaml`) are moved to `docs/architecture/*.bak` pending deprecation; use Helmfile for all new changes.
