# Python Shared Libraries

This package houses shared Python modules consumed by the domain monoliths and supporting tooling.

## Directory conventions
- `clients/` – HTTP/gRPC/Kafka clients generated from contracts and wrapped with resilience primitives (timeouts, backoff, circuit breakers).
- `db/` – database session management, migrations helpers, and repositories.
- `http/` – ASGI middlewares, rate limiting, caching, and request utilities.
- `kafka/` – producer/consumer base classes with schema validation.
- `config/` – 12-factor configuration helpers leveraging environment variables and secret stores.
- `logging/` – structured logging utilities and OpenTelemetry instrumentation helpers.
- `schemas/` – dataclasses and Pydantic models generated from contracts.

## Packaging
- Implemented as a Poetry/pyproject package for installation in each service.
- Will include type hints and enforce style via Ruff/Black/MyPy in CI.
- Publish internally to a package index as part of the build pipeline.

