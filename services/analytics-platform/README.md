# Analytics Platform Domain

## Purpose
The analytics platform consolidates all analytical workloads into a modular monolith, enabling shared pipelines, reusable feature engineering, and unified APIs.

## Modules (planned)
- `core` – common domain models, data access, and shared utilities imported from `libs/python`.
- `forecasting` – time-series forecasting, retraining workflows, feature engineering.
- `risk` – risk analytics, scenario analysis, and reporting.
- `research` – research APIs including arbitrage, transition, renewables, and satellite intelligence.
- `market` – commodities, market insights, and marketplace integrations.
- `quantum` – quantum optimizer and advanced modeling.
- `api` – external REST/GraphQL interfaces with versioned contracts.
- `jobs` – background workers and orchestration tasks.

## Status
- [x] Directory scaffolded
- [ ] Import shared analytics utilities
- [ ] Migrate service modules into internal packages
- [ ] Publish consolidated contracts and tests

## Migration considerations
- Maintain backward compatibility of public APIs via versioned routes and feature flags.
- Preserve long-running training jobs; ensure idempotent orchestration with retries.
- Align data schemas and storage migrations alongside service consolidation.

## Migration Status

- [x] Directory scaffolded
- [x] Shared commodity feature engineering moved to `libs/python/src/carbon254/analytics/commodity_features.py`
- [ ] Move forecasting routers from `platform/apps/analytics-service`
- [ ] Move ML research APIs from `platform/apps/ml-service`
- [ ] Align data models and repositories with shared DB package

## Dependencies

- Requires `carbon254-libs-python`
- Uses contracts defined under `contracts/http/analytics-platform` (TBD)
- Consumes event schemas from `contracts/events`

