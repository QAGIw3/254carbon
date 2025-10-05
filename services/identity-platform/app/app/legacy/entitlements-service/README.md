# Entitlements Service

Centralized entitlement and permission management for the 254Carbon platform.

## Overview

The Entitlements Service manages user access control for markets, products, and channels. It checks tenant permissions for instruments and validates data access rights.

## Features

- **Entitlement Checks**: Validate user access to instruments and channels
- **Bulk Operations**: Check multiple entitlements efficiently
- **User Entitlements**: List all entitlements for a user
- **Channel-based Access**: Support for hub, API, downloads, and stream channels
- **Tenant Isolation**: Enforce tenant-based access control

## API Endpoints

### `POST /entitlements/check`
Check if user has entitlement for instrument and channel.

**Request**:
```json
{
  "user_id": "user-123",
  "tenant_id": "tenant-456",
  "instrument_id": "MISO.LMP.INDIANA.HUB",
  "channel": "api"
}
```

**Response**:
```json
{
  "entitled": true,
  "reason": null
}
```

### `POST /entitlements/bulk-check`
Check multiple entitlements in a single request.

**Request**:
```json
{
  "user_id": "user-123",
  "tenant_id": "tenant-456",
  "checks": [
    {"instrument_id": "MISO.LMP.INDIANA.HUB", "channel": "api"},
    {"instrument_id": "CAISO.LMP.SP15", "channel": "stream"}
  ]
}
```

**Response**:
```json
{
  "results": [
    {"instrument_id": "MISO.LMP.INDIANA.HUB", "channel": "api", "entitled": true},
    {"instrument_id": "CAISO.LMP.SP15", "channel": "stream", "entitled": false}
  ]
}
```

### `GET /entitlements/user/{user_id}?tenant_id={tenant_id}`
Get all entitlements for a user.

**Response**:
```json
{
  "user_id": "user-123",
  "tenant_id": "tenant-456",
  "entitlements": [
    {
      "market": "power",
      "product": "lmp",
      "channels": {"hub": true, "api": true, "downloads": true, "stream": true},
      "from_date": "2025-01-01",
      "to_date": null
    }
  ]
}
```

### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T12:00:00Z",
  "database_connected": true
}
```

## Channels

- **hub**: Web Hub UI access
- **api**: REST API access
- **downloads**: File download access
- **stream**: WebSocket/SSE streaming access

## Configuration

Environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `PORT`: Service port (default: `8011`)

## Database Schema

### `pg.entitlement_product`

```sql
CREATE TABLE pg.entitlement_product (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    market VARCHAR(50) NOT NULL,
    product VARCHAR(50) NOT NULL,
    channels JSONB NOT NULL,
    from_date DATE,
    to_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Channels JSONB format**:
```json
{
  "hub": true,
  "api": true,
  "downloads": false,
  "stream": true
}
```

## Usage by Other Services

### Python Example

```python
import httpx

async def check_user_access(user_id: str, tenant_id: str, instrument_id: str, channel: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://entitlements-service:8011/entitlements/check",
            json={
                "user_id": user_id,
                "tenant_id": tenant_id,
                "instrument_id": instrument_id,
                "channel": channel
            },
            timeout=5.0
        )
        data = response.json()
        return data["entitled"]
```

## Deployment

### Docker

```bash
docker build -t 254carbon/entitlements-service:latest .
docker run -p 8011:8011 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  254carbon/entitlements-service:latest
```

### Kubernetes

See `k8s/deployment.yaml` for Kubernetes deployment configuration.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8011

# Test
curl -X POST http://localhost:8011/entitlements/check \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user-123","tenant_id":"tenant-456","instrument_id":"MISO.LMP.INDIANA.HUB","channel":"api"}'
```

## Architecture

```
┌─────────────────┐
│   PostgreSQL    │
│  (Entitlements) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Entitlements   │
│    Service      │
│  (Port 8011)    │
└────────┬────────┘
         │ Access Checks
         ▼
┌─────────────────┐
│  API Gateway    │
│  Streaming Svc  │
│  Other Services │
└─────────────────┘
```

