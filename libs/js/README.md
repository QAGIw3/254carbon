# JavaScript/TypeScript Shared Libraries

This package provides shared TypeScript modules used by the frontend applications and any Node-based services.

## Directory conventions
- `clients/` – REST/GraphQL clients generated from contracts with resilient fetch wrappers.
- `http/` – middleware, interceptors, and utilities for retries, backoff, and circuit-breaking.
- `logging/` – structured logging utilities compatible with the observability domain.
- `schemas/` – TypeScript types and Zod validators generated from OpenAPI/GraphQL/event contracts.

## Packaging
- Distributed as a pnpm workspace package; also publishable to an internal npm registry.
- TypeScript-first with strict `tsconfig` enforcing consistent conventions.
- Prettier and ESLint configurations will live under `.tools/` and be shared across apps.

