# Service Inventory and Domain Mapping

This document captures the current microservices and assigns each to a target domain modular monolith as outlined in the consolidation plan. It serves as a checkpoint before moving code. This inventory will drive the scaffolding of the new `services/` layout and identify shared components for `libs/`.

## Edge Gateway Domain
- `platform/apps/api-gateway`
- `platform/apps/gateway`
- `platform/apps/graphql-gateway`
- `platform/apps/intelligence-gateway`
- `platform/apps/download-center` (customer-facing edge download endpoints)

## Analytics Platform Domain
- `platform/apps/analytics-service`
- `platform/apps/ml-service`
- `platform/apps/battery-analytics`
- `platform/apps/climate-risk`
- `platform/apps/gas_coal_analytics`
- `platform/apps/market-insights`
- `platform/apps/quantum-optimizer`
- `platform/apps/risk-service`
- `platform/apps/signals-service` (analytics output APIs)
- `platform/apps/satellite-intel`
- `platform/apps/realtime-forecast-service`
- `platform/apps/lmp-decomposition-service`
- `platform/apps/routing-service`
- `platform/apps/backtesting-service`
- `platform/apps/fundamental-models`
- `platform/apps/fundamentals-consumer`
- `platform/apps/report-service`
- `platform/apps/scenario-engine`
- `platform/apps/curve-service`
- `platform/apps/hydrogen-service`
- `platform/apps/marketplace`
- `platform/apps/ppa-workbench`
- `platform/apps/commodities-service`
- `platform/apps/ai-service`
- `platform/apps/intelligence-gateway` (analytics assist interfaces)

## Data Platform Domain
- `platform/apps/streaming-service`
- `platform/apps/stream-processing`
- `platform/apps/data-quality-service`
- `platform/apps/signals-service` (data feed component)
- `platform/data/connectors/*`
- `platform/data/ingestion-orch/*`
- `platform/data/reference/*`
- `platform/data/schemas/*`
- `platform/tests/load/*`

## Identity Platform Domain
- `platform/apps/auth-service`
- `platform/apps/entitlements-service`
- `platform/apps/gateway/auth.py` (logic to be migrated)

## Observability Domain
- `platform/apps/metrics-service`
- `platform/infra/monitoring/*`
- `platform/apps/backtesting-service/metrics.py` (move to shared metrics tooling)

## Frontend & External Apps
- `platform/apps/web-hub`
- `platform/apps/excel-addin`
- `sdk/python`

## Shared Utilities (Target: libs)
- `platform/shared/*`
- `platform/apps/api-gateway/clients/*`
- `platform/apps/streaming-service/clients/*`
- `libs/python/src/carbon254/analytics/commodity_features.py`
- `platform/apps/api-gateway/db.py`
- `platform/apps/entitlements-service/db.py`
- `platform/apps/streaming-service/db.py`

## Infrastructure Artifacts
- Root level `*-deployment.yaml` manifests (target removal after Helm alignment)
- `platform/infra/*` (source of truth)
- `platform/ci/*`
- `Makefile` and `platform/docker-compose.yml`

## Notes
- Some services appear in multiple domains (e.g., `signals-service`) because they span analytics outputs and streaming ingestion. Detailed module splits will be decided during consolidation.
- Commodity feature engineering now lives under `libs/python/src/carbon254/analytics/commodity_features.py` and should not be re-implemented in services.
- The inventory intentionally keeps SDKs and frontend apps outside the domain monoliths but ensures they consume the new shared libraries and contracts.

