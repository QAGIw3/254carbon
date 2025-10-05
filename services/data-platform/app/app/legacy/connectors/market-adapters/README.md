# Market Adapters

Market-specific logic and endpoints for the 254Carbon platform.

## Overview

Market adapters provide specialized functionality for different energy markets (MISO, CAISO, PJM, ERCOT, etc.). Each adapter is independently developed and deployed, allowing market teams to iterate without affecting the core platform.

## Structure

```
market-adapters/
├── common/           # Shared base classes and utilities
├── miso/            # MISO-specific endpoints and logic
├── caiso/           # CAISO compliance and reporting
├── ercot/           # ERCOT features (future)
└── pjm/             # PJM features (future)
```

## Adapter Development Guide

### Creating a New Adapter

1. **Create adapter directory**:
   ```bash
   mkdir -p platform/data/connectors/market-adapters/my-market/
   ```

2. **Implement endpoints**:
   ```python
   # my-market/endpoints.py
   from fastapi import APIRouter
   from common.adapter_base import MarketAdapter
   
   router = APIRouter(prefix="/api/v1/my-market", tags=["My Market"])
   
   @router.get("/trading-summary")
   async def get_trading_summary():
       # Implementation
       pass
   ```

3. **Register with API Gateway**:
   ```python
   # In API Gateway main.py
   from platform.data.connectors.market_adapters.my_market.endpoints import router
   app.include_router(router)
   ```

### Base Adapter Class

All adapters should extend `MarketAdapter` for consistency:

```python
from common.adapter_base import MarketAdapter

class MISOAdapter(MarketAdapter):
    def __init__(self):
        super().__init__(market_name="MISO")
    
    async def validate_entitlement(self, user: dict, resource: str):
        # Market-specific entitlement logic
        pass
```

## Available Adapters

### MISO Adapter

**Location**: `miso/`

**Features**:
- Trading position summaries
- Daily trading reports
- Portfolio risk analysis
- Price alerts
- Congestion analysis
- Trading opportunities

**Endpoints**:
- `GET /api/v1/miso/trading-summary`
- `GET /api/v1/miso/daily-report`
- `GET /api/v1/miso/portfolio-risk`
- `GET /api/v1/miso/price-alerts`
- `POST /api/v1/miso/price-alerts`
- `GET /api/v1/miso/congestion-analysis`
- `GET /api/v1/miso/trading-opportunities`

### CAISO Adapter

**Location**: `caiso/`

**Features**:
- Settlement data reports
- Resource adequacy reports
- Renewable portfolio reports
- Compliance requirements
- Penalty calculations
- Batch report generation

**Endpoints**:
- `POST /api/v1/caiso/compliance/reports/settlement`
- `POST /api/v1/caiso/compliance/reports/resource-adequacy`
- `POST /api/v1/caiso/compliance/reports/renewable-portfolio`
- `GET /api/v1/caiso/compliance/requirements/current`
- `GET /api/v1/caiso/compliance/penalties/calculator`
- `POST /api/v1/caiso/compliance/reports/batch`

## Best Practices

### 1. Separation of Concerns

Keep market-specific logic isolated:
- ✅ Market-specific calculations in adapter
- ✅ Market-specific data models in adapter
- ❌ Don't mix market logic with core platform code

### 2. Consistent API Design

Follow platform API conventions:
- Use standard HTTP methods (GET, POST, PUT, DELETE)
- Return consistent error formats
- Include proper authentication/authorization
- Document with OpenAPI/Swagger

### 3. Entitlement Checking

Always validate user entitlements:
```python
from auth import verify_token
from entitlements import check_entitlement

@router.get("/endpoint")
async def endpoint(user=Depends(verify_token)):
    await check_entitlement(user, "market", "power", "api")
    # ... implementation
```

### 4. Database Access

Use connection pooling:
```python
from db import get_postgres_pool, get_clickhouse_client

async def get_data():
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        # Query database
        pass
```

### 5. Error Handling

Provide meaningful error messages:
```python
try:
    # ... implementation
except Exception as e:
    logger.error(f"Error in endpoint: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### 6. Metrics Tracking

Track adapter usage:
```python
from metrics import track_request

@router.get("/endpoint")
async def endpoint():
    track_request("miso_trading_summary")
    # ... implementation
```

## Testing

### Unit Tests

```python
import pytest
from miso.endpoints import get_trading_summary

@pytest.mark.asyncio
async def test_trading_summary():
    result = await get_trading_summary(date="2025-10-05")
    assert result.total_positions >= 0
```

### Integration Tests

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_miso_endpoint():
    response = client.get("/api/v1/miso/trading-summary")
    assert response.status_code == 200
```

## Deployment

Market adapters are deployed as part of the API Gateway:

1. Adapters are imported as Python modules
2. Routers are included in the main FastAPI app
3. No separate deployment needed
4. Changes require API Gateway restart

## Migration from Gateway

When migrating endpoints from the old Gateway:

1. Move endpoint file to appropriate adapter directory
2. Update imports to use absolute paths
3. Keep router structure intact
4. Update tests to reflect new location
5. Update documentation

## Future Enhancements

- **Separate Deployments**: Deploy each adapter as independent service
- **Plugin Architecture**: Load adapters dynamically at runtime
- **Versioning**: Support multiple API versions per adapter
- **Rate Limiting**: Adapter-specific rate limits
- **Caching**: Adapter-specific cache strategies

