# New Services API Guide

Comprehensive guide for newly implemented 254Carbon platform services.

## Table of Contents

1. [LMP Decomposition Service](#lmp-decomposition-service)
2. [Trading Signals Service](#trading-signals-service)
3. [Marketplace Service](#marketplace-service)
4. [Transformer ML Models](#transformer-ml-models)
5. [New Market Connectors](#new-market-connectors)

---

## LMP Decomposition Service

**Base URL:** `http://lmp-decomposition-service:8009`

### Overview

The LMP Decomposition Service breaks down nodal Locational Marginal Prices (LMP) into three components:
- **Energy Component**: Base energy cost
- **Congestion Component**: Transmission constraint costs
- **Loss Component**: Transmission loss costs

**Formula:** `LMP = Energy + Congestion + Loss`

### Endpoints

#### 1. Decompose LMP

**POST** `/api/v1/lmp/decompose`

Decompose nodal LMP into components for specified nodes and time range.

**Request:**
```json
{
  "node_ids": ["PJM.HUB.WEST", "PJM.WESTERN"],
  "start_time": "2025-10-04T00:00:00Z",
  "end_time": "2025-10-04T01:00:00Z",
  "iso": "PJM"
}
```

**Response:**
```json
[
  {
    "timestamp": "2025-10-04T00:00:00Z",
    "node_id": "PJM.HUB.WEST",
    "lmp_total": 50.25,
    "energy_component": 45.00,
    "congestion_component": 3.50,
    "loss_component": 1.75,
    "currency": "USD",
    "unit": "MWh"
  }
]
```

#### 2. Calculate PTDF

**POST** `/api/v1/lmp/ptdf`

Calculate Power Transfer Distribution Factor between nodes.

**Request:**
```json
{
  "source_node": "PJM.HUB.WEST",
  "sink_node": "PJM.EASTERN",
  "constraint_id": "WEST_TO_EAST",
  "iso": "PJM"
}
```

**Response:**
```json
{
  "source_node": "PJM.HUB.WEST",
  "sink_node": "PJM.EASTERN",
  "constraint_id": "WEST_TO_EAST",
  "ptdf_value": 0.75,
  "interpretation": "1 MW injection at PJM.HUB.WEST causes 0.75 MW flow on WEST_TO_EAST"
}
```

#### 3. Basis Surface

**POST** `/api/v1/lmp/basis-surface`

Calculate hub-to-node basis risk metrics.

**Request:**
```json
{
  "hub_id": "PJM.HUB.WEST",
  "node_ids": ["PJM.WESTERN", "PJM.EASTERN"],
  "as_of_date": "2025-10-04",
  "iso": "PJM"
}
```

**Response:**
```json
{
  "hub_id": "PJM.HUB.WEST",
  "as_of_date": "2025-10-04",
  "basis_surface": [
    {
      "node_id": "PJM.WESTERN",
      "mean_basis": 2.5,
      "std_basis": 1.8,
      "percentile_95": 5.2,
      "percentile_5": -0.5,
      "correlation_to_hub": 0.92
    }
  ]
}
```

---

## Trading Signals Service

**Base URL:** `http://signals-service:8016`

### Overview

Generates algorithmic trading signals using multiple strategies with backtesting capabilities.

### Strategies Available

- **mean_reversion**: Buy when price below MA, sell when above
- **momentum**: Follow trending markets
- **spread_trading**: Trade spreads between related markets
- **volatility**: Trade volatility regimes
- **ml_ensemble**: Combine multiple strategies

### Endpoints

#### 1. Generate Signal

**POST** `/api/v1/signals/generate`

Generate trading signal using specified strategy.

**Request:**
```json
{
  "strategy": "mean_reversion",
  "instrument_id": "PJM.HUB.WEST",
  "market_data": {
    "price": 50.0,
    "prices": [45.0, 46.0, 48.0, 50.0, 52.0, 51.0]
  }
}
```

**Response:**
```json
{
  "signal_id": "SIG-20251004-120000",
  "instrument_id": "PJM.HUB.WEST",
  "signal_type": "SELL",
  "strength": "moderate",
  "confidence": 0.75,
  "entry_price": 50.0,
  "target_price": 46.0,
  "stop_loss": 53.0,
  "strategy": "mean_reversion",
  "generated_at": "2025-10-04T12:00:00Z",
  "expires_at": "2025-10-05T12:00:00Z",
  "rationale": "Price 2.1 std above 20-day MA. Mean reversion expected."
}
```

#### 2. Backtest Strategy

**POST** `/api/v1/signals/backtest`

Backtest trading strategy on historical data.

**Request:**
```json
{
  "strategy": "momentum",
  "instruments": ["PJM.HUB.WEST", "MISO.HUB.INDIANA"],
  "start_date": "2025-07-01T00:00:00Z",
  "end_date": "2025-09-30T00:00:00Z",
  "initial_capital": 100000.0,
  "position_size_pct": 0.1
}
```

**Response:**
```json
{
  "strategy": "momentum",
  "total_return": 15.5,
  "sharpe_ratio": 1.25,
  "max_drawdown": -8.5,
  "win_rate": 0.58,
  "total_trades": 150,
  "avg_trade_duration_hours": 36.5,
  "best_trade": 12.3,
  "worst_trade": -5.8
}
```

#### 3. Send FIX Order

**POST** `/api/v1/signals/fix/order`

Send order via FIX protocol to trading platform.

**Request:**
```json
{
  "order_id": "ORD-001",
  "instrument_id": "PJM.HUB.WEST",
  "side": "BUY",
  "quantity": 100.0,
  "order_type": "LIMIT",
  "price": 50.0,
  "time_in_force": "DAY"
}
```

---

## Marketplace Service

**Base URL:** `http://marketplace:8015`

### Overview

Third-party data integration platform for revenue sharing with external data providers.

### Endpoints

#### 1. Register Partner

**POST** `/api/v1/marketplace/partners/register`

Register as a data provider partner.

**Request:**
```json
{
  "company_name": "Weather Insights Co.",
  "contact_name": "Jane Smith",
  "email": "jane@weatherinsights.com",
  "website": "https://weatherinsights.com",
  "description": "Weather impact forecasting for energy markets",
  "data_products": ["weather_forecasts", "irradiance_data"]
}
```

**Response:**
```json
{
  "partner_id": "PTR-20251004-120000",
  "company_name": "Weather Insights Co.",
  "status": "pending",
  "api_key": "sk_live_123456",
  "revenue_share_pct": 70.0,
  "registered_date": "2025-10-04T12:00:00Z"
}
```

#### 2. List Products

**GET** `/api/v1/marketplace/products`

List available data products in marketplace.

**Query Parameters:**
- `category` (optional): Filter by category (market_data, analytics, forecasts, alerts)
- `partner_id` (optional): Filter by partner

**Response:**
```json
[
  {
    "product_id": "PRD-WEATHER-001",
    "partner_id": "PTR-WEATHERCO",
    "name": "Weather Impact Forecasts",
    "description": "ML-powered weather impact on power prices",
    "category": "forecasts",
    "pricing_model": "subscription",
    "monthly_subscription": 299.99,
    "free_tier_calls": 100,
    "documentation_url": "https://docs.254carbon.ai/partners/weather",
    "status": "active"
  }
]
```

#### 3. Create Sandbox

**POST** `/api/v1/marketplace/sandbox`

Create sandbox environment for testing.

**Request:**
```json
{
  "partner_id": "PTR-20251004-120000",
  "product_id": "PRD-WEATHER-001"
}
```

**Response:**
```json
{
  "sandbox_id": "SBX-20251004-120000",
  "partner_id": "PTR-20251004-120000",
  "product_id": "PRD-WEATHER-001",
  "api_key": "sk_test_654321",
  "endpoint": "https://sandbox.254carbon.ai/api/v1/",
  "rate_limit": "100 calls/hour",
  "expires_at": "2025-11-03T12:00:00Z"
}
```

---

## Transformer ML Models

**Service:** ML Service  
**Base URL:** `http://ml-service:8006`

### Overview

State-of-the-art transformer models for price forecasting with uncertainty quantification.

### Features

- **Multi-head Attention**: Captures complex temporal dependencies
- **Positional Encoding**: Preserves time-series ordering
- **Monte Carlo Dropout**: Provides prediction intervals
- **Ensemble Methods**: Combines multiple model predictions

### Usage

#### Train Transformer Model

**POST** `/api/v1/ml/train`

**Request:**
```json
{
  "instrument_ids": ["PJM.HUB.WEST"],
  "start_date": "2024-01-01",
  "end_date": "2025-09-30",
  "model_type": "transformer",
  "hyperparameters": {
    "hidden_size": 256,
    "num_layers": 6,
    "num_heads": 8,
    "dropout": 0.1,
    "learning_rate": 0.0001,
    "max_epochs": 50
  }
}
```

#### Generate Forecast

**POST** `/api/v1/ml/forecast`

**Request:**
```json
{
  "instrument_id": "PJM.HUB.WEST",
  "horizon_months": 12,
  "model_version": null
}
```

**Response:**
```json
{
  "instrument_id": "PJM.HUB.WEST",
  "model_version": "20251004_120000",
  "forecasts": [
    {
      "month_ahead": 1,
      "forecast_price": 48.5,
      "ci_lower": 45.2,
      "ci_upper": 51.8
    }
  ],
  "confidence_intervals": [...]
}
```

#### Train Multimodal Transformer

Enable cross-commodity, multimodal forecasting across related instruments.

**POST** `/api/v1/ml/train`

**Request:**
```json
{
  "instrument_ids": [
    "POWER.NYISO.ZONEA",
    "GAS.HENRY_HUB.MONTH_AHEAD"
  ],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "model_type": "multimodal_transformer",
  "hyperparameters": {
    "seq_len": 128,
    "forecast_horizons": [7, 30, 90],
    "batch_size": 16,
    "epochs": 40,
    "learning_rate": 0.0004,
    "d_model": 320,
    "num_heads": 8,
    "num_layers": 4
  }
}
```

The service automatically aligns price, fundamentals, and weather series based on the multimodal mapping configuration (`config/multimodal_mapping.yaml`). Metrics are logged per commodity and persisted with the model registry metadata.

#### Forecast with Multimodal Transformer

**POST** `/api/v1/ml/forecast`

**Request:**
```json
{
  "instrument_id": "POWER.NYISO.ZONEA",
  "model_version": "20250218_153045"
}
```

**Response:**
```json
{
  "instrument_id": "POWER.NYISO.ZONEA",
  "model_version": "20250218_153045",
  "forecasts": [
    {"month_ahead": 7, "forecast_price": 68.4, "std": 3.1},
    {"month_ahead": 30, "forecast_price": 71.9, "std": 4.7},
    {"month_ahead": 90, "forecast_price": 74.2, "std": 6.0}
  ],
  "confidence_intervals": [
    {"lower": 62.3, "upper": 74.5},
    {"lower": 62.7, "upper": 81.1},
    {"lower": 62.3, "upper": 86.1}
  ],
  "extras": {
    "fusion_gates": {
      "price": 0.52,
      "fundamentals": 0.31,
      "weather": 0.17
    },
    "commodity_key": "commodity_a",
    "commodity_order": ["commodity_a", "commodity_b"],
    "cross_attention": [[0.64, 0.36], [0.28, 0.72]]
  }
}
```

`extras` contains modality fusion weights and averaged cross-commodity attention useful for diagnostics and explainability.

---

## New Market Connectors

### IESO (Ontario, Canada)

**Market:** Ontario Electricity Market  
**Product:** HOEP (Hourly Ontario Energy Price)  
**Frequency:** Hourly  
**Currency:** CAD  

**Instruments:**
- `IESO.HOEP.ON` - Hourly Ontario Energy Price
- `IESO.DEMAND.ON` - Ontario demand
- `IESO.GENERATION.{FUEL}.ON` - Generation by fuel type

### NEM (Australia)

**Market:** National Electricity Market  
**Product:** Spot Price (30-minute)  
**Frequency:** 30 minutes  
**Currency:** AUD  

**Regions:**
- NSW (New South Wales)
- QLD (Queensland)
- SA (South Australia)
- TAS (Tasmania)
- VIC (Victoria)

**Instruments:**
- `NEM.SPOT.{REGION}` - Regional spot prices
- `NEM.DEMAND.{REGION}` - Regional demand
- `NEM.INTERCONNECTOR.{NAME}` - Inter-regional flows

### Brazil ONS

**Market:** Brazilian Power Market  
**Product:** PLD (Preço de Liquidação das Diferenças)  
**Frequency:** Weekly  
**Currency:** BRL  

**Subsystems:**
- N (North)
- NE (Northeast)
- S (South)
- SECO (Southeast/Midwest)

**Instruments:**
- `ONS.PLD.{SUBSYSTEM}` - Settlement prices
- `ONS.LOAD_FORECAST.{SUBSYSTEM}` - Load forecasts
- `ONS.GENERATION.{SOURCE}.{SUBSYSTEM}` - Generation by source
- `ONS.RESERVOIR.{NAME}` - Hydro reservoir levels

---

## Performance Characteristics

### Service Latencies (p95)

| Service | Endpoint | Target | Typical |
|---------|----------|--------|---------|
| LMP Decomposition | `/api/v1/lmp/decompose` | <300ms | 180ms |
| ML Forecast | `/api/v1/ml/forecast` | <1000ms | 650ms |
| Trading Signals | `/api/v1/signals/generate` | <200ms | 120ms |
| Marketplace | `/api/v1/marketplace/products` | <150ms | 80ms |

### Data Freshness

| Connector | Update Frequency | Max Staleness |
|-----------|-----------------|---------------|
| IESO HOEP | Hourly | 2 hours |
| NEM Spot | 30 minutes | 90 minutes |
| Brazil PLD | Weekly | 1 week |

---

## Authentication & Security

All services support Keycloak OIDC authentication. Include Bearer token in Authorization header:

```bash
Authorization: Bearer <token>
```

Marketplace partner operations require partner API key:

```bash
X-Partner-Key: sk_live_123456
```

---

## Examples

### Python Example - LMP Decomposition

```python
import requests

url = "http://lmp-decomposition-service:8009/api/v1/lmp/decompose"
payload = {
    "node_ids": ["PJM.HUB.WEST"],
    "start_time": "2025-10-04T00:00:00Z",
    "end_time": "2025-10-04T01:00:00Z",
    "iso": "PJM"
}

response = requests.post(url, json=payload)
components = response.json()

for comp in components:
    print(f"Node: {comp['node_id']}")
    print(f"  Total LMP: ${comp['lmp_total']:.2f}/MWh")
    print(f"  Energy: ${comp['energy_component']:.2f}")
    print(f"  Congestion: ${comp['congestion_component']:.2f}")
    print(f"  Loss: ${comp['loss_component']:.2f}")
```

### Python Example - Trading Signals

```python
import requests
import numpy as np

# Generate sample price history
prices = list(45 + np.random.randn(50) * 5)

url = "http://signals-service:8016/api/v1/signals/generate"
payload = {
    "strategy": "mean_reversion",
    "instrument_id": "PJM.HUB.WEST",
    "market_data": {
        "price": prices[-1],
        "prices": prices
    }
}

response = requests.post(url, json=payload)
signal = response.json()

print(f"Signal: {signal['signal_type']}")
print(f"Confidence: {signal['confidence']:.2%}")
print(f"Entry: ${signal['entry_price']:.2f}")
print(f"Target: ${signal.get('target_price', 'N/A')}")
print(f"Rationale: {signal['rationale']}")
```

### cURL Example - Marketplace

```bash
# List products
curl -X GET http://marketplace:8015/api/v1/marketplace/products

# Register partner
curl -X POST http://marketplace:8015/api/v1/marketplace/partners/register \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Data Provider Inc.",
    "contact_name": "Alice Johnson",
    "email": "alice@dataprovider.com",
    "description": "Premium market data provider",
    "data_products": ["real_time_prices"]
  }'

# Get marketplace analytics
curl -X GET http://marketplace:8015/api/v1/marketplace/analytics
```

---

## Error Handling

All services return standard HTTP status codes:

- **200 OK**: Request successful
- **400 Bad Request**: Invalid parameters
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation error
- **500 Internal Server Error**: Server error

Error responses include descriptive messages:

```json
{
  "detail": "No trained model found for PJM.HUB.WEST"
}
```

---

## Rate Limits

| Service | Authenticated | Unauthenticated |
|---------|--------------|-----------------|
| LMP Decomposition | 100 req/min | 10 req/min |
| Trading Signals | 50 req/min | 5 req/min |
| Marketplace | 100 req/min | 20 req/min |
| ML Forecast | 20 req/min | 5 req/min |

---

## Support

- **Documentation**: https://docs.254carbon.ai
- **API Issues**: api-support@254carbon.ai
- **Marketplace**: marketplace@254carbon.ai
- **GitHub**: https://github.com/254carbon/market-intelligence/issues

---

**Last Updated:** October 4, 2025  
**API Version:** 1.0.0
