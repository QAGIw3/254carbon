# ML Service API Extensions – Refining & Renewables

## Refining Router (`/api/v1/refining`)

- `POST /crack-optimize`
  - Inputs: region, as_of_date, crack_types, crude_code (default WTI), optional price overrides.
  - Response: calculated crack spread metrics per configuration and persisted row count.
- `POST /refinery-yields`
  - Inputs: region, as_of_date, crude_type, optional process constraints and price overrides.
  - Response: expected yields, margin metrics, persistence count.
- `POST /demand-elasticity`
  - Inputs: product, price instrument, fundamentals variable, optional cross-product configuration.
  - Response: elasticity calculations (own and optional cross) and persistence count.
- `POST /fuel-substitution`
  - Inputs: region, fundamentals entity, EV adoption parameters, optional infrastructure assumptions.
  - Response: substitution metrics (elasticity, EV impact, infrastructure utilisation) with persistence count.

### Example – Crack Optimisation

```json
POST /api/v1/refining/crack-optimize
{
  "as_of_date": "2024-05-15",
  "region": "PADD3",
  "crack_types": ["3:2:1", "5:3:2"],
  "crude_code": "OIL.WTI",
  "model_version": "v1"
}
```

## Renewables Router (`/api/v1/renewables`)

- `POST /rin-forecast`
  - Inputs: forecast horizon, RIN categories (D4/D5/D6), optional overrides.
  - Response: forecast summary and persistence count.
- `POST /biodiesel-spread`
  - Inputs: as_of_date, optional region, incentive overrides.
  - Response: spread statistics, arbitrage metrics, persistence count.
- `POST /carbon-intensity`
  - Inputs: fuel_type, pathway, transport assumptions.
  - Response: lifecycle CI breakdown, persistence count.
- `POST /policy-impact`
  - Inputs: list of policy scenarios (RIN demand/supply shifts, incentive changes).
  - Response: RIN/biodiesel impact summaries, persistence count.

### Example – RIN Forecast

```json
POST /api/v1/renewables/rin-forecast
{
  "as_of_date": "2024-05-15",
  "categories": ["D4", "D5", "D6"],
  "horizon_days": 60,
  "model_version": "v1"
}
```

Metrics: both routers expose Prometheus counters (`ml_service_refining_api_requests_total`, `ml_service_renewables_api_requests_total`) and histograms (`*_latency_seconds`) labelled by endpoint.

