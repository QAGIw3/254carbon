# 254Carbon Microservices Architecture

## Overview

The 254Carbon platform has been refactored from a monolithic Gateway into a microservices architecture with clear separation of concerns. This document describes the new architecture and migration path.

## Architecture Diagram

```
                                    ┌─────────────────┐
                                    │   Keycloak      │
                                    │   (OIDC)        │
                                    └────────┬────────┘
                                             │
                         ┌───────────────────┼───────────────────┐
                         │                   │                   │
                         ▼                   ▼                   ▼
┌──────────────┐   ┌──────────┐      ┌──────────┐       ┌──────────┐
│   Clients    │   │   Auth   │      │Entitle-  │       │ Metrics  │
│ (Web/Mobile) │   │ Service  │      │ ments    │       │ Service  │
└──────┬───────┘   │(Port 8010│      │ Service  │       │(Port 8012│
       │           └─────┬────┘      │(Port 8011│       └─────┬────┘
       │                 │           └─────┬────┘             │
       │                 │                 │                  │
       │                 └─────────┬───────┘                  │
       │                           │                          │
       ▼                           ▼                          ▼
┌─────────────┐             ┌─────────────┐          ┌──────────────┐
│   Ingress   │             │             │          │  Prometheus  │
│   (Nginx)   │             │             │          │  (Scraping)  │
└──────┬──────┘             │             │          └──────────────┘
       │                    │             │
       ├────/api/*─────────▶│             │
       │                    │             │
       ▼                    │ Internal    │
┌─────────────┐             │ Service     │
│API Gateway  │◀────────────│ Mesh        │
│(Port 8000)  │             │             │
│ Stateless   │             │             │
└──────┬──────┘             │             │
       │                    │             │
       ├────/ws/*──────────▶│             │
       ├────/sse/*─────────▶│             │
       │                    │             │
       ▼                    ▼             │
┌─────────────┐      ┌─────────────┐     │
│ Streaming   │      │Market       │     │
│ Service     │      │Adapters     │     │
│(Port 8001)  │      │(MISO, CAISO)│     │
│ Stateful    │      └─────────────┘     │
└──────┬──────┘                           │
       │                                  │
       ▼                                  ▼
┌─────────────────────────────────────────────┐
│                                             │
│  Data Layer                                 │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐    │
│  │ ClickHouse│ │PostgreSQL│ │  Redis  │    │
│  └──────────┘ └──────────┘ └─────────┘    │
│                                             │
│  ┌──────────┐                               │
│  │  Kafka   │                               │
│  └──────────┘                               │
└─────────────────────────────────────────────┘
```

## Services

### 1. API Gateway (Port 8000)

**Type**: Stateless REST API

**Responsibilities**:
- Core REST endpoints (instruments, ticks, curves, fundamentals)
- Authentication via Auth Service
- Authorization via Entitlements Service
- Rate limiting (slowapi)
- Redis caching with adaptive TTL
- Request routing to market adapters
- Metrics tracking to Metrics Service

**Scaling**: Horizontal (3+ replicas)

**Resources**: 512Mi-2Gi memory, 500m-2000m CPU

**Endpoints**:
- `GET /api/v1/instruments`
- `GET /api/v1/prices/ticks`
- `GET /api/v1/curves/forward`
- `GET /api/v1/fundamentals`
- `GET /api/v1/cache/stats`
- `POST /api/v1/cache/warm`
- `GET /api/v1/miso/*` (market adapters)
- `POST /api/v1/caiso/compliance/*` (market adapters)

### 2. Streaming Service (Port 8001)

**Type**: Stateful streaming

**Responsibilities**:
- WebSocket connections for real-time streaming
- Server-Sent Events (SSE) for HTTP streaming
- Kafka consumer integration
- Connection lifecycle management
- Subscription filtering (instrument, commodity, wildcard)
- Real-time price fanout

**Scaling**: Horizontal with sticky sessions (2-3 replicas)

**Resources**: 2Gi-8Gi memory, 1000m-4000m CPU

**Endpoints**:
- `WebSocket /ws/stream`
- `GET /sse/stream`
- `GET /health`

### 3. Auth Service (Port 8010)

**Type**: Internal microservice

**Responsibilities**:
- JWT token verification via Keycloak
- JWKS caching (1 hour TTL)
- User claims extraction
- WebSocket token validation

**Scaling**: Horizontal (2+ replicas)

**Resources**: 256Mi-512Mi memory, 250m-500m CPU

**Endpoints**:
- `POST /auth/verify`
- `POST /auth/verify-ws`
- `GET /auth/user-info`
- `POST /auth/refresh-keys`

### 4. Entitlements Service (Port 8011)

**Type**: Internal microservice

**Responsibilities**:
- User entitlement validation
- Market/product/channel access control
- Bulk entitlement checks
- Tenant-based permissions

**Scaling**: Horizontal (2+ replicas)

**Resources**: 256Mi-512Mi memory, 250m-500m CPU

**Endpoints**:
- `POST /entitlements/check`
- `POST /entitlements/bulk-check`
- `GET /entitlements/user/{user_id}`

### 5. Metrics Service (Port 8012)

**Type**: Internal microservice

**Responsibilities**:
- Centralized metrics collection
- Prometheus export (`/metrics`)
- Request counting by service/endpoint
- Latency tracking
- Connection tracking

**Scaling**: Horizontal (2+ replicas)

**Resources**: 256Mi-512Mi memory, 250m-500m CPU

**Endpoints**:
- `GET /metrics` (Prometheus format)
- `POST /metrics/track`
- `POST /metrics/latency`
- `POST /metrics/connection`

### 6. Market Adapters

**Type**: Module (not separate service)

**Location**: `platform/data/connectors/market-adapters/`

**Adapters**:
- **MISO**: Trading summaries, reports, risk, alerts, congestion, opportunities
- **CAISO**: Settlement reports, resource adequacy, renewable portfolio, compliance

**Integration**: Imported by API Gateway as routers

## Service Communication

### Authentication Flow

```
1. Client → API Gateway (with JWT in Authorization header)
2. API Gateway → Auth Service (POST /auth/verify)
3. Auth Service → Keycloak (JWKS verification)
4. Auth Service → API Gateway (user claims)
5. API Gateway → Client (authorized response)
```

### Entitlement Check Flow

```
1. API Gateway → Entitlements Service (POST /entitlements/check)
2. Entitlements Service → PostgreSQL (query permissions)
3. Entitlements Service → API Gateway (entitled: true/false)
4. API Gateway → Client (403 if not entitled, 200 if entitled)
```

### Metrics Tracking Flow

```
1. API Gateway/Streaming Service → Metrics Service (POST /metrics/track)
2. Metrics Service → Internal metrics store
3. Prometheus → Metrics Service (GET /metrics)
4. Prometheus → Metrics database
```

## Rate Limiting

Implemented in API Gateway using `slowapi`:

| Tier | Limit | Applies To |
|------|-------|------------|
| Public | 100/min | Health, public endpoints |
| Authenticated | 1000/min | Standard API calls |
| Heavy | 10/min | Large data queries |
| Cache Write | 5/min | Cache warming |

## Deployment

### Prerequisites

- Kubernetes cluster (local or production)
- PostgreSQL database
- ClickHouse database
- Redis cluster
- Kafka cluster
- Keycloak OIDC provider

### Deploy All Services

```bash
# Deploy all new microservices
kubectl apply -f all-services-deployment.yaml

# Update ingress routing
kubectl apply -f ingress-updated.yaml

# Verify deployments
kubectl get pods -n market-intelligence

# Check service health
kubectl port-forward -n market-intelligence svc/api-gateway 8000:8000
curl http://localhost:8000/health

kubectl port-forward -n market-intelligence svc/streaming-service 8001:8001
curl http://localhost:8001/health
```

### Individual Service Deployment

```bash
# Auth Service
kubectl apply -f platform/apps/auth-service/k8s/deployment.yaml

# Entitlements Service
kubectl apply -f platform/apps/entitlements-service/k8s/deployment.yaml

# Metrics Service
kubectl apply -f platform/apps/metrics-service/k8s/deployment.yaml

# API Gateway
kubectl apply -f platform/apps/api-gateway/k8s/deployment.yaml

# Streaming Service
kubectl apply -f platform/apps/streaming-service/k8s/deployment.yaml
```

## Migration from Old Gateway

### Breaking Changes

1. **WebSocket Endpoint**: `/api/v1/stream` → `ws://streaming-service:8001/ws/stream`
2. **SSE Endpoint**: `/api/v1/stream/sse` → `http://streaming-service:8001/sse/stream`
3. **Service Discovery**: Services now call other services via HTTP (not local imports)

### Backward Compatibility

- REST API endpoints remain unchanged (`/api/v1/*`)
- Market adapter endpoints remain unchanged (`/api/v1/miso/*`, `/api/v1/caiso/*`)
- Authentication flow remains JWT-based
- Ingress routes old paths to new services transparently

### Migration Strategy

**Phase 1**: Deploy new services (completed ✓)
- Auth Service
- Entitlements Service
- Metrics Service

**Phase 2**: Deploy new gateways (completed ✓)
- API Gateway (coexists with old gateway)
- Streaming Service

**Phase 3**: Update ingress (completed ✓)
- Route `/api/*` to API Gateway
- Route `/ws/*` and `/sse/*` to Streaming Service

**Phase 4**: Client migration
- Update WebSocket clients to use new URL
- Update SSE clients to use new URL
- Test thoroughly in staging

**Phase 5**: Deprecate old gateway
- Monitor metrics for old gateway usage
- Gradually reduce old gateway replicas
- Remove old gateway after validation period

## Benefits

### 1. Independent Scaling

- **API Gateway**: Scale for request throughput
- **Streaming Service**: Scale for connection count
- **Auth/Entitlements**: Scale for auth load

### 2. Fault Isolation

- Streaming issues don't affect REST API
- Auth issues can be handled independently
- Services can fail without cascading

### 3. Technology Flexibility

- Services can use different languages/frameworks
- Easy to rewrite individual services
- Polyglot architecture support

### 4. Team Autonomy

- Market teams can iterate on adapters independently
- Core platform team owns gateways
- Auth/Entitlements can evolve separately

### 5. Operational Excellence

- Better monitoring per service
- Easier debugging (service-specific logs)
- Faster deployments (deploy only changed services)

## Monitoring & Observability

### Metrics

All services export metrics to Metrics Service:
- Request counts
- Latency histograms
- Error rates
- Active connections

### Logs

Structured logging per service:
- Service name tag
- Request ID tracing
- Error stack traces
- Performance metrics

### Health Checks

Every service exposes `/health`:
- Liveness probe (is service running?)
- Readiness probe (can service handle traffic?)
- Dependency checks (database, cache, etc.)

## Security

### Authentication

- JWT validation via Auth Service
- Keycloak OIDC integration
- Token expiration checks
- JWKS caching for performance

### Authorization

- Entitlement checks via Entitlements Service
- Tenant-based isolation
- Channel-based access control (hub, api, downloads, stream)
- Market/product/instrument level permissions

### Network Security

- Internal services not exposed to internet
- Ingress controls external access
- Service-to-service authentication (optional mTLS)

## Performance

### API Gateway

- **Throughput**: 10,000+ req/s per replica
- **Latency**: P95 < 100ms
- **Cache Hit Rate**: 70-90% (depending on TTL)

### Streaming Service

- **Connections**: 5,000+ per replica
- **Message Latency**: < 100ms (Kafka to client)
- **Memory**: ~2KB per connection

### Microservices

- **Auth Service**: < 10ms P95
- **Entitlements Service**: < 20ms P95 (with cache)
- **Metrics Service**: < 5ms P95 (async)

## Troubleshooting

### API Gateway not responding

```bash
# Check API Gateway pods
kubectl get pods -n market-intelligence -l app=api-gateway

# Check logs
kubectl logs -n market-intelligence -l app=api-gateway --tail=100

# Check Auth Service (dependency)
kubectl get pods -n market-intelligence -l app=auth-service
```

### WebSocket connections failing

```bash
# Check Streaming Service pods
kubectl get pods -n market-intelligence -l app=streaming-service

# Check logs
kubectl logs -n market-intelligence -l app=streaming-service --tail=100

# Check Kafka (dependency)
kubectl get pods -n market-intelligence -l app=kafka
```

### Authentication failures

```bash
# Check Auth Service
kubectl logs -n market-intelligence -l app=auth-service --tail=100

# Check Keycloak connectivity
kubectl exec -it -n market-intelligence <auth-pod> -- curl http://keycloak:8080/auth/realms/254carbon
```

## Future Enhancements

1. **Service Mesh**: Implement Istio/Linkerd for mTLS and advanced traffic management
2. **API Versioning**: Support multiple API versions simultaneously
3. **GraphQL Gateway**: Unified GraphQL interface over all services
4. **Event-Driven**: Implement event bus for async service communication
5. **Multi-Region**: Deploy services across multiple regions
6. **A/B Testing**: Traffic splitting for gradual feature rollouts
7. **Circuit Breakers**: Automatic failure handling and recovery
8. **Distributed Tracing**: OpenTelemetry integration for request tracing

## Documentation

Each service has its own README:

- [Auth Service README](platform/apps/auth-service/README.md)
- [Entitlements Service README](platform/apps/entitlements-service/README.md)
- [Metrics Service README](platform/apps/metrics-service/README.md)
- [API Gateway README](platform/apps/api-gateway/README.md)
- [Streaming Service README](platform/apps/streaming-service/README.md)
- [Market Adapters README](platform/data/connectors/market-adapters/README.md)

