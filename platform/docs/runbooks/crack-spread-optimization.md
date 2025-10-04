# Crack Spread Optimisation Runbook

## Purpose
- Generate refinery crack spread analytics (3:2:1, 5:3:2 etc.).
- Persist outputs to ClickHouse (`market_intelligence.refining_crack_optimization`).
- Provide downstream access via ML service API and GraphQL (`ch.refining_crack_optimization`).

## Data Inputs
- Prices: `OIL.RBOB`, `OIL.ULSD`, `OIL.JET`, crude benchmarks (`OIL.WTI`, `OIL.BRENT`).
- Fundamentals: gasoline/diesel demand for elasticity and substitution metrics (`ch.fundamentals_series`).
- Quality checks performed via `DataQualityFramework` thresholds (refined products & RIN domains).

## Execution Paths
1. **API Trigger** – `POST /api/v1/refining/crack-optimize` with region, crack types, optional constraints.
2. **Batch Job** – `run_daily_refining_jobs()` located in `platform/apps/ml-service/jobs/refining_jobs.py` (invokes crack optimisation, refinery yields, elasticity & substitution in sequence).

### Sample Job Invocation
```python
from platform.apps.ml_service.jobs import run_daily_refining_jobs

run_daily_refining_jobs()
```

## Outputs
- Stored metrics: crack spread value, margin per barrel, optimal yields JSON, applied constraints.
- Diagnostics snapshot persisted for auditing (yield slate, price context).
- GraphQL: `crackOptimizationResults(region, start, end, limit)`.

## Operational Notes
- Ensure price feeds aligned to `spot` price_type; ingestion normalised in `EIAOpenDataConnector`.
- Runtimes: typical batch < 10s for default regions.
- Monitoring: Prometheus counter/histogram (`ml_service_refining_api_*`).
- Troubleshooting: check ClickHouse insert failures and DQ warnings (`market_intelligence.data_quality_issues`).

