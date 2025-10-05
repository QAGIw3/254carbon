# Metrics Service

Centralized metrics collection and Prometheus export for the 254Carbon platform.

## Overview

The Metrics Service collects metrics from all microservices and exports them in Prometheus format. It provides a unified view of platform performance and health.

## Features

- **Request Tracking**: Count requests by service, endpoint, method, and status
- **Latency Monitoring**: Track request duration across services
- **Connection Tracking**: Monitor active connections (WebSocket, HTTP, etc.)
- **Prometheus Export**: Standard `/metrics` endpoint for Prometheus scraping
- **Multi-Service Support**: Collect metrics from all platform services

## API Endpoints

### `GET /metrics`
Prometheus metrics endpoint (text format).

**Response** (Prometheus format):
```
# HELP service_requests_total Total requests across all services
# TYPE service_requests_total counter
service_requests_total{service="api-gateway",endpoint="/api/v1/instruments",method="GET",status="200"} 1234.0
service_requests_total{service="streaming-service",endpoint="/ws/stream",method="WEBSOCKET",status="101"} 56.0

# HELP service_request_duration_seconds Request latency across all services
# TYPE service_request_duration_seconds histogram
service_request_duration_seconds_bucket{service="api-gateway",endpoint="/api/v1/instruments",le="0.005"} 100.0
service_request_duration_seconds_bucket{service="api-gateway",endpoint="/api/v1/instruments",le="0.01"} 250.0
...
```

### `POST /metrics/track`
Track an API request.

**Request**:
```json
{
  "endpoint": "/api/v1/instruments",
  "method": "GET",
  "status": 200,
  "service": "api-gateway"
}
```

**Response**:
```json
{
  "status": "tracked"
}
```

### `POST /metrics/latency`
Track request latency.

**Request**:
```json
{
  "endpoint": "/api/v1/instruments",
  "duration_seconds": 0.125,
  "service": "api-gateway"
}
```

**Response**:
```json
{
  "status": "tracked"
}
```

### `POST /metrics/connection`
Track connection changes.

**Request**:
```json
{
  "connection_type": "websocket",
  "delta": 1,
  "service": "streaming-service"
}
```

Note: `delta` is +1 for new connection, -1 for closed connection.

**Response**:
```json
{
  "status": "tracked"
}
```

### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T12:00:00Z",
  "metrics_count": 0
}
```

## Metrics Types

### Counters
- `service_requests_total`: Total requests by service, endpoint, method, status

### Histograms
- `service_request_duration_seconds`: Request latency distribution
- `stream_latency_seconds`: Streaming latency distribution

### Gauges
- `service_active_connections`: Current active connections by service and type

## Configuration

Environment variables:

- `PORT`: Service port (default: `8012`)

## Usage by Other Services

### Python Example

```python
import httpx
import time

async def track_api_request(endpoint: str, method: str, status: int):
    async with httpx.AsyncClient() as client:
        await client.post(
            "http://metrics-service:8012/metrics/track",
            json={
                "endpoint": endpoint,
                "method": method,
                "status": status,
                "service": "my-service"
            },
            timeout=1.0  # Fire-and-forget, short timeout
        )

async def track_request_with_latency(endpoint: str):
    start_time = time.time()
    
    try:
        # ... do work ...
        duration = time.time() - start_time
        
        # Track latency
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://metrics-service:8012/metrics/latency",
                json={
                    "endpoint": endpoint,
                    "duration_seconds": duration,
                    "service": "my-service"
                },
                timeout=1.0
            )
    except Exception:
        pass  # Don't fail the request if metrics fail
```

### Decorator Pattern

```python
import functools
import time
import httpx

def track_latency(endpoint: str, service: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                try:
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            "http://metrics-service:8012/metrics/latency",
                            json={
                                "endpoint": endpoint,
                                "duration_seconds": duration,
                                "service": service
                            },
                            timeout=1.0
                        )
                except Exception:
                    pass  # Ignore metrics errors
        return wrapper
    return decorator

@track_latency("/api/v1/instruments", "api-gateway")
async def get_instruments():
    # ... implementation ...
    pass
```

## Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: '254carbon-metrics'
    static_configs:
      - targets: ['metrics-service:8012']
    scrape_interval: 15s
```

## Deployment

### Docker

```bash
docker build -t 254carbon/metrics-service:latest .
docker run -p 8012:8012 254carbon/metrics-service:latest
```

### Kubernetes

See `k8s/deployment.yaml` for Kubernetes deployment configuration.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8012

# View metrics
curl http://localhost:8012/metrics

# Track a request
curl -X POST http://localhost:8012/metrics/track \
  -H "Content-Type: application/json" \
  -d '{"endpoint":"/test","method":"GET","status":200,"service":"test-service"}'
```

## Architecture

```
┌─────────────────┐
│   Prometheus    │
│   (Scraping)    │
└────────┬────────┘
         │ /metrics
         ▼
┌─────────────────┐
│ Metrics Service │
│  (Port 8012)    │
└────────┬────────┘
         │ POST /metrics/track
         ▼
┌─────────────────┐
│  API Gateway    │
│  Streaming Svc  │
│  Other Services │
└─────────────────┘
```

## Best Practices

1. **Fire-and-Forget**: Use short timeouts (1-2s) when tracking metrics to avoid slowing down requests
2. **Error Handling**: Don't fail requests if metrics tracking fails
3. **Async Tracking**: Track metrics asynchronously to minimize latency impact
4. **Batch Tracking**: For high-throughput services, consider batching metrics
5. **Label Cardinality**: Be careful with high-cardinality labels (e.g., user IDs)

