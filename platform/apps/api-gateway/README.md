# API Gateway

Stateless REST API gateway for the 254Carbon platform.

## Overview

The API Gateway provides unified access to all platform services through a RESTful API. It handles authentication, authorization, rate limiting, caching, and routing to backend services and market adapters.

## Features

- **Stateless Design**: No WebSocket or streaming connections (use Streaming Service)
- **Rate Limiting**: Configurable limits per endpoint and user tier
- **Authentication**: JWT validation via Auth Service
- **Authorization**: Entitlement checks via Entitlements Service
- **Caching**: Redis-based caching with adaptive TTL
- **Market Adapters**: Integrated MISO and CAISO endpoints
- **Metrics**: Request tracking to Metrics Service
- **Horizontal Scaling**: Stateless design allows easy scaling

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│API Gateway  │────▶│ Auth Service │
│  (Port 8000)│     └──────────────┘
└──────┬──────┘
       │
       ├────────────▶┌──────────────────┐
       │             │Entitlements Svc  │
       │             └──────────────────┘
       │
       ├────────────▶┌──────────────────┐
       │             │ Metrics Service  │
       │             └──────────────────┘
       │
       └────────────▶┌──────────────────┐
                     │  ClickHouse      │
                     │  PostgreSQL      │
                     │  Redis           │
                     └──────────────────┘
```

## API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /api/v1/instruments` - List instruments
- `GET /api/v1/prices/ticks` - Historical price ticks
- `GET /api/v1/curves/forward` - Forward curves
- `GET /api/v1/fundamentals` - Fundamentals time series

### Cache Management

- `GET /api/v1/cache/stats` - Cache statistics
- `POST /api/v1/cache/warm` - Warm cache

### Market Adapters

- `GET /api/v1/miso/*` - MISO endpoints
- `POST /api/v1/caiso/compliance/*` - CAISO compliance

## Rate Limits

| Tier | Limit | Applies To |
|------|-------|------------|
| Public | 100/min | Unauthenticated endpoints |
| Authenticated | 1000/min | Standard API calls |
| Heavy | 10/min | Large data queries |
| Cache Write | 5/min | Cache warming operations |

## Configuration

Environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `CLICKHOUSE_HOST`: ClickHouse hostname
- `REDIS_URL`: Redis connection string
- `AUTH_SERVICE_URL`: Auth Service URL (default: `http://auth-service:8010`)
- `ENTITLEMENTS_SERVICE_URL`: Entitlements Service URL
- `METRICS_SERVICE_URL`: Metrics Service URL
- `ENVIRONMENT`: Environment (development/production)

## Deployment

### Docker

```bash
docker build -t 254carbon/api-gateway:latest .
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e CLICKHOUSE_HOST=clickhouse \
  254carbon/api-gateway:latest
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000

# Test
curl http://localhost:8000/health
```

## Testing

```bash
# Unit tests
pytest tests/

# Load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## Performance

- **Horizontal Scaling**: Add replicas to handle more requests
- **Caching**: Redis cache reduces database load
- **Connection Pooling**: Efficient database connections
- **Rate Limiting**: Protects against abuse
- **Async**: Full async/await for high concurrency

## Monitoring

Metrics exported to Metrics Service:
- Request count by endpoint
- Request latency
- Error rates
- Cache hit rates

## Migration from Old Gateway

The API Gateway replaces the stateless parts of the old Gateway:
- ✅ REST endpoints (instruments, prices, curves, etc.)
- ✅ Authentication and authorization
- ✅ Caching and rate limiting
- ❌ WebSocket streaming (moved to Streaming Service)
- ❌ SSE streaming (moved to Streaming Service)

## Related Services

- **Streaming Service**: WebSocket/SSE streaming (port 8001)
- **Auth Service**: Authentication (port 8010)
- **Entitlements Service**: Authorization (port 8011)
- **Metrics Service**: Metrics collection (port 8012)

