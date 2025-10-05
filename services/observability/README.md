# Observability Domain

## Purpose
The observability domain consolidates metrics, logging, and tracing components into a dedicated service that can be consumed by all domains.

## Modules (planned)
- `metrics` – Prometheus exporters, metrics ingestion, and aggregation.
- `logging` – structured logging pipeline, sinks, and log shipping controls.
- `tracing` – OpenTelemetry collector configuration and instrumentation helpers.
- `dashboards` – curated dashboards and alert definitions.
- `clients` – libraries for emitting telemetry to the observability stack.

## Status
- [x] Directory scaffolded
- [ ] Migrate metrics-service code and shared exporters
- [ ] Define telemetry interfaces and instrumentations
- [ ] Publish dashboards and alert runbooks

## Migration considerations
- Keep metrics names stable; provide relabeling during transition.
- Ensure log formats remain backward compatible for downstream SIEM or analytics tools.
- Run load testing on new collectors/exporters to size resource requirements.

## Migration Status

- [x] Directory scaffolded
- [ ] Move metrics-service exporters into `metrics/`
- [ ] Provide shared logging/tracing instrumentation via `carbon254.logging`
- [ ] Define dashboards and alert rules under `infra/monitoring`
- [ ] Ensure contracts for telemetry endpoints are published

## Dependencies

- Relies on OpenTelemetry collectors and Prometheus deployed via Helm
- Exposes configuration consumed by other domains through `carbon254.logging`
- Dashboards and alerts maintained alongside code in `infra/monitoring`

