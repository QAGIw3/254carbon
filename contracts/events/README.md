# Event Contracts

This directory stores versioned schemas for event-based integrations (Kafka, CDC, etc.).

## Conventions
- Each topic has its own folder (e.g., `kafka/<topic-name>`).
- Schemas are authored in Avro or JSON Schema and validated in CI.
- Backwards-compatible changes (additive) are enforced via schema compatibility checks.

## Next Steps
- Import existing schemas from `platform/data/schemas`.
- Define schema lints and validation in CI pipeline.

