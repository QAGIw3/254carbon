# RIN Price Forecasting Runbook

## Purpose
- Forecast Renewable Identification Number (RIN) prices for D4/D5/D6 categories.
- Capture scenario-sensitive policy impacts and biodiversity incentives.
- Persist results to ClickHouse (`market_intelligence.rin_price_forecast`, `market_intelligence.renewables_policy_impact`).

## Data Inputs
- Price history: `RIN.D4`, `RIN.D5`, `RIN.D6` (spot prices).
- Diesel/Biodiesel prices for spread analysis: `OIL.ULSD`, `BIO.BIODIESEL`.
- Policy scenarios (demand/supply adjustments, incentive deltas).

## Execution Paths
1. **API Trigger** – `POST /api/v1/renewables/rin-forecast` with categories & horizon.
2. **Batch Job** – `run_daily_renewables_jobs()` in `platform/apps/ml-service/jobs/renewables_jobs.py` (executes forecasts, spread metrics, carbon intensity, policy impact).

### Sample Job Invocation
```python
from platform.apps.ml_service.jobs import run_daily_renewables_jobs

run_daily_renewables_jobs()
```

## Outputs
- Forecast curve per category with horizon, standard deviation proxy, driver metadata.
- Policy impact scenarios covering RFS/LCFS adjustments (`renewables_policy_impacts` GraphQL field).
- Diagnostics: regression fit summarised in persisted JSON (drivers / volatility).

## Monitoring & Validation
- Prometheus metrics: `ml_service_renewables_api_requests_total`, `ml_service_renewables_api_latency_seconds`.
- Insert validation: verify ClickHouse row counts post-run (`SELECT count(*) FROM ch.rin_price_forecast WHERE as_of_date = today()`).
- Alerting: DQ service monitors RIN price bounds (DataQualityFramework `rins` domain).

