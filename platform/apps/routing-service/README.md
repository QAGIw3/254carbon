# Multi-Source Redundancy Router

**Intelligent routing between multiple upstream data sources with automatic failover, blending, and trust scoring.**

## Overview

The Multi-Source Redundancy Router continuously chooses, blends, or fails over between multiple upstream data sources (e.g., carbon intensity feeds, LMP endpoints, registry metadata) to maximize availability, correctness, continuity, and auditability—without manual intervention.

## Key Features

- **Observability-first**: Every routing decision is explainable, reproducible, and auditable
- **Incremental enforcement**: Start in shadow (observe-only), then advisory, then active routing, finally autonomous failover + blending
- **Deterministic + confidence-scored**: Same inputs ⇒ same decision, with a confidence metric
- **Non-blocking**: Degraded or failed upstream never stalls downstream consumers (graceful degradation mode)

## Architecture Components

### 1. Trust Scoring Engine

Computes trust scores for data sources based on:
- **Freshness**: How recent is the data relative to SLA
- **Error rate**: Reliability of responses  
- **Deviation**: Consistency with peer consensus
- **Consistency**: Low variance over time
- **Uptime**: Historical availability

### 2. Routing Policy Engine

Core routing decision algorithm:
1. Filter sources by trust and freshness thresholds
2. Single source if only one candidate
3. Blend if dispersion is low
4. Remove outliers (MAD) and retry if dispersion is high
5. Synthetic fallback if no good candidates

### 3. Circuit Breaker

Automatic source removal from rotation after N consecutive critical failures with cooldown period.

### 4. Health Monitoring

Track per-source metrics:
- Freshness lag, latency, error rate, completeness
- Deviation from consensus baseline
- Anomaly detection flags

## Quick Start

### Installation

```bash
# Install dependencies
cd platform/apps/routing-service
pip install -r requirements.txt

# Initialize database schema
psql -h localhost -U market_user -d market_intelligence -f ../../data/schemas/postgres/routing_schema.sql
```

### Running the Service

```bash
# Development mode
python main.py

# Production with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8007 --workers 4
```

### Docker Deployment

```bash
# Build image
docker build -t 254carbon/routing-service:latest .

# Run container
docker run -p 8007:8007 \
  -e POSTGRES_HOST=postgresql \
  -e POSTGRES_DB=market_intelligence \
  254carbon/routing-service:latest
```

### Kubernetes Deployment

```bash
kubectl apply -f ../../../routing-service-deployment.yaml
```

## API Endpoints

### Route Metric Request

```http
POST /api/v1/routing/route
Content-Type: application/json

{
  "metric_key": "carbon_intensity.us_east",
  "candidate_values": [
    {
      "source_id": "source_a",
      "value": 450.2,
      "timestamp": "2025-01-15T10:00:00Z"
    },
    {
      "source_id": "source_b",
      "value": 452.1,
      "timestamp": "2025-01-15T10:00:05Z"
    }
  ],
  "mode": "routed"
}
```

**Response:**
```json
{
  "value": 451.15,
  "timestamp": "2025-01-15T10:00:10Z",
  "strategy": "blend",
  "confidence": 0.92,
  "sources": [
    {
      "source_id": "source_a",
      "value": 450.2,
      "trust_score": 0.89,
      "weight": 0.52,
      "freshness_lag_sec": 60
    },
    {
      "source_id": "source_b",
      "value": 452.1,
      "trust_score": 0.85,
      "weight": 0.48,
      "freshness_lag_sec": 65
    }
  ],
  "decision_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_synthetic": false
}
```

### Update Health Metrics

```http
POST /api/v1/routing/health
Content-Type: application/json

{
  "source_id": "source_a",
  "metric_key": "carbon_intensity.us_east",
  "freshness_lag_sec": 60,
  "response_latency_ms": 150,
  "error_rate_win": 0.02,
  "completeness_pct": 99.5,
  "deviation_from_blend": 0.005,
  "last_value": 450.2
}
```

### Get Sources Health

```http
GET /api/v1/routing/sources/health?metric_key=carbon_intensity.us_east
```

### Get Routing Decisions

```http
GET /api/v1/routing/decisions?metric_key=carbon_intensity.us_east&limit=100
```

### Reload Policy

```http
POST /api/v1/routing/policy/reload
```

## Configuration

Routing policy is stored in the `pg.routing_policy` table. Default configuration:

```yaml
policy_version: v1
thresholds:
  min_trust: 0.55
  max_fresh_lag_sec: 180
  stable_dispersion: 0.012
  switch_margin: 0.07
weights:
  freshness: 0.30
  error_rate: 0.20
  deviation: 0.15
  consistency: 0.15
  uptime: 0.20
synthetic:
  decay_factor: 0.85
  max_consecutive: 12
hysteresis:
  min_intervals_before_switch: 3
mad_k_factor: 3.0
```

## Database Schema

Key tables:
- `pg.source_registry` - Source metadata and configuration
- `pg.source_health` - Time-series health metrics
- `pg.trust_scores` - Computed trust scores over time
- `pg.routing_decisions` - Audit trail of routing decisions
- `pg.routing_policy` - Versioned policy configuration
- `pg.circuit_breaker_state` - Circuit breaker states
- `pg.routing_overrides` - Manual override rules

## Testing

```bash
# Run unit tests
cd tests
pytest test_engine.py -v

# Run with coverage
pytest test_engine.py --cov=../engine --cov-report=html
```

## Monitoring

### Prometheus Metrics

The service exposes standard FastAPI metrics plus custom metrics:
- `routing_decisions_total{strategy, metric_key}` - Total decisions by strategy
- `routing_confidence_avg{metric_key}` - Average confidence scores
- `synthetic_fallback_ratio{metric_key}` - Percentage of synthetic decisions
- `source_trust_score{source_id, metric_key}` - Current trust scores
- `circuit_breaker_state{source_id}` - Circuit breaker states

### Grafana Dashboards

Import the dashboard configuration from `monitoring/grafana/routing-service-dashboard.json`:
- Trust score time series
- Switch frequency
- Dispersion spikes  
- Synthetic fallback ratio
- Decision latency

### Alert Rules

Key alerts:
- `SyntheticFallbackRatioHigh` - Synthetic fallback ratio > 20% over 5 minutes
- `SourceTrustScoreLow` - Source trust score < 0.4 for 10 minutes
- `CircuitBreakerOpen` - Circuit breaker open for critical source
- `RoutingDecisionLatencyHigh` - p95 latency > 100ms

## Integration with Gateway

The routing service integrates with the gateway API to provide transparent source routing:

```python
# In gateway endpoint
from routing_service_client import RoutingClient

routing_client = RoutingClient(base_url="http://routing-service:8007")

@app.get("/api/v1/metrics/{metric_key}")
async def get_metric(metric_key: str, mode: str = "routed"):
    # Fetch from multiple sources
    source_values = await fetch_from_sources(metric_key)
    
    if mode == "routed":
        # Use routing service
        decision = await routing_client.route(metric_key, source_values)
        return {
            "value": decision.value,
            "confidence": decision.confidence,
            "sources": decision.sources
        }
    else:
        # Direct passthrough
        return source_values[0]
```

## Audit Integration

All routing decisions are logged to the audit trail:

```python
await AuditLogger.log_access(
    user_id=user_id,
    tenant_id=tenant_id,
    action="routing_decision",
    resource_type="metric",
    resource_id=metric_key,
    request=request,
    success=True,
    details={
        "strategy": decision.strategy,
        "confidence": decision.confidence,
        "sources": [s["source_id"] for s in decision.sources]
    }
)
```

## Incremental Rollout

### Phase 0: Design & Instrumentation
- ✅ Add health emission hooks in existing connectors
- ✅ Stand up source_health table + basic freshness computations

### Phase 1: Shadow Mode  
- Compute trust scores + route decisions but DO NOT alter downstream endpoints
- Store decisions separately
- Compare shadow decisions vs currently used single source

### Phase 2: Advisory Mode
- Expose API flag `?mode=advisory` returning suggested routed_value alongside current_value
- Alert on scenarios where advisory would have avoided stale data

### Phase 3: Active Single Selection
- Enable automatic switching between sources (no blending yet)
- Apply hysteresis (min_hold_intervals)

### Phase 4: Weighted Blending & Synthetic Fallback
- Activate blending for dispersion <= threshold
- Implement synthetic fallback with explicit metrics

### Phase 5: Advanced Autonomy
- Add ML-based trust predictor & dynamic threshold tuning
- License-aware blending and cost-aware source selection

## License

Proprietary - 254Carbon Market Intelligence Platform

## Support

For issues or questions:
- Internal: #routing-service Slack channel
- Email: platform-team@254carbon.ai
