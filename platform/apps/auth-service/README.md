# Auth Service

Centralized authentication service for the 254Carbon platform.

## Overview

The Auth Service provides JWT token verification and user claims extraction for all microservices. It integrates with Keycloak OIDC for authentication.

## Features

- **JWT Verification**: Validates tokens using Keycloak's JWKS
- **User Claims Extraction**: Returns normalized user information
- **WebSocket Auth**: Specialized endpoint for WebSocket token validation
- **JWKS Caching**: Caches public keys for 1 hour to reduce Keycloak load
- **Health Checks**: Monitors Keycloak connectivity

## API Endpoints

### `POST /auth/verify`
Verify a JWT token for REST API usage.

**Request**:
```json
{
  "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response**:
```json
{
  "valid": true,
  "user_claims": {
    "sub": "user-123",
    "email": "user@example.com",
    "name": "John Doe",
    "tenant_id": "tenant-456",
    "roles": ["trader", "analyst"],
    "scopes": ["read:ticks", "write:orders"],
    "groups": ["MISO_traders"]
  }
}
```

### `POST /auth/verify-ws`
Verify a JWT token for WebSocket connections.

Same request/response format as `/auth/verify`.

### `GET /auth/user-info?token={token}`
Get user information from token.

**Response**:
```json
{
  "user_id": "user-123",
  "email": "user@example.com",
  "name": "John Doe",
  "tenant_id": "tenant-456",
  "roles": ["trader", "analyst"]
}
```

### `POST /auth/refresh-keys`
Force refresh of JWKS cache.

**Response**:
```json
{
  "status": "success",
  "keys_count": 2,
  "timestamp": "2025-10-05T12:00:00Z"
}
```

### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T12:00:00Z",
  "keycloak_reachable": true
}
```

## Configuration

Environment variables:

- `KEYCLOAK_URL`: Keycloak realm URL (default: `http://keycloak:8080/auth/realms/254carbon`)
- `KEYCLOAK_AUDIENCE`: Expected audience claim (default: `market-intelligence-api`)
- `PORT`: Service port (default: `8010`)

## Usage by Other Services

### Python Example

```python
import httpx

async def verify_user_token(token: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://auth-service:8010/auth/verify",
            json={"token": token},
            timeout=5.0
        )
        data = response.json()
        
        if data["valid"]:
            return data["user_claims"]
        else:
            raise ValueError(f"Invalid token: {data.get('error')}")
```

### Node.js Example

```javascript
const axios = require('axios');

async function verifyUserToken(token) {
    const response = await axios.post(
        'http://auth-service:8010/auth/verify',
        { token },
        { timeout: 5000 }
    );
    
    if (response.data.valid) {
        return response.data.user_claims;
    } else {
        throw new Error(`Invalid token: ${response.data.error}`);
    }
}
```

## Deployment

### Docker

```bash
docker build -t 254carbon/auth-service:latest .
docker run -p 8010:8010 \
  -e KEYCLOAK_URL=http://keycloak:8080/auth/realms/254carbon \
  254carbon/auth-service:latest
```

### Kubernetes

See `k8s/deployment.yaml` for Kubernetes deployment configuration.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8010

# Test
curl -X POST http://localhost:8010/auth/verify \
  -H "Content-Type: application/json" \
  -d '{"token": "YOUR_JWT_TOKEN"}'
```

## Architecture

```
┌─────────────────┐
│   Keycloak      │
│   (OIDC)        │
└────────┬────────┘
         │ JWKS
         ▼
┌─────────────────┐
│  Auth Service   │
│  (Port 8010)    │
└────────┬────────┘
         │ Token Validation
         ▼
┌─────────────────┐
│  API Gateway    │
│  Streaming Svc  │
│  Other Services │
└─────────────────┘
```

## Security

- Tokens are validated using RSA-256 signatures
- JWKS is cached for 1 hour (configurable)
- Expired tokens are rejected
- Audience and issuer claims are strictly validated
- No tokens are stored or logged in full

