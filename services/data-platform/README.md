# Data Platform Domain

## Purpose
The data platform unifies ingestion, streaming, and data-quality workflows under a cohesive service boundary, enabling shared schema validation and consistent delivery to downstream analytics.

## Modules (planned)
- `ingestion` – connectors, ingestion orchestrators, and partner integrations.
- `streaming` – real-time processing, Kafka consumers/producers, stream enrichment.
- `quality` – data quality framework, rules, and monitoring.
- `storage` – partitioned data management and access utilities.
- `api` – APIs for ingestion management, schemas, and operational controls.
- `jobs` – scheduled batches and maintenance tasks.

## Status
- [x] Directory scaffolded
- [ ] Centralize schemas and validation logic
- [ ] Migrate streaming and connector code
- [ ] Establish operational runbooks and alerts

## Migration considerations
- Ensure every connector has a compatibility-tested schema in `contracts/events`.
- Provide migration scripts for existing Kafka topics or ingestion pipelines.
- Implement backpressure handling and circuit breakers for external data sources.

## Migration Status

- [x] Directory scaffolded
- [ ] Move streaming-service entrypoints and clients
- [ ] Move stream-processing jobs and shared utilities
- [ ] Relocate data-quality-service DAGs and rules
- [ ] Import connector schemas from `platform/data/connectors`

## Dependencies

- Depends on shared Kafka tooling from `carbon254.kafka`
- Schemas defined under `contracts/events`
- Requires access to connector credentials managed via Vault

