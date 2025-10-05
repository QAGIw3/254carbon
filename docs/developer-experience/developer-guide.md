# Developer Guide

## Repository Layout

- `services/` – domain modular monoliths (edge-gateway, analytics-platform, data-platform, identity-platform, observability)
- `platform/apps/` – legacy microservices; migrate functionality into `services/`
- `libs/python/`, `libs/js/` – shared packages for clients, HTTP, analytics, logging
- `contracts/` – OpenAPI, GraphQL, and event schemas (source of truth)
- `docs/` – architecture notes, ADRs, runbooks, developer experience docs
- `infra/` – Helmfile, Helm charts, Terraform, Argo CD configurations

## Working with Shared Libraries

- Python consumers depend on `carbon254-libs-python` (Poetry project under `libs/python`).
- TypeScript consumers depend on `@carbon254/libs-js` (pnpm/npm workspace under `libs/js`).
- Add new shared modules under `libs/python/src/carbon254/` or `libs/js/src/`; export via `__all__` or `index.ts`.
- Run `poetry build` or `pnpm build` to verify packaging before publishing.

## Contracts

- Author OpenAPI specs under `contracts/http/<domain>/`.
- Place shared schemas in `contracts/http/shared/` and reference via `$ref`.
- GraphQL SDL lives under `contracts/graphql/<domain>/`.
- Event schemas reside under `contracts/events/`; organize by topic and version.
- CI should validate contracts and regenerate clients.

## Running Services Locally

- Use `platform/docker-compose.yml` for dependencies (ClickHouse, Kafka, PostgreSQL, Keycloak).
- Each service under `services/<domain>` includes a `pyproject.toml` (Python) or package manifest; use Poetry/pnpm for development.
- FastAPI apps expose `/health`; contract-driven tests ensure compatibility.
- Use `make up` (planned) for orchestrating local stacks.

## Migration Guidelines

1. Update `docs/architecture/service-inventory.md` when moving functionality.
2. Move shared utilities into `libs/` before relocating service modules.
3. Maintain OpenAPI/GraphQL parity; deprecate old endpoints with headers.
4. Add telemetry hooks via `carbon254.logging` and `carbon254.http` middleware.
5. Remove legacy code only after the new domain path is stable and documented.

## Tooling

- Pre-commit hooks and CI templates will live under `.tools/` (to be populated).
- Linters: Ruff/Black/MyPy for Python, ESLint/Prettier for TypeScript.
- Testing: pytest, Vitest/Jest; coverage thresholds enforced in CI.

## Support

- Architecture questions: consult ADRs under `docs/adr/`.
- Contract changes: raise design review to ensure backwards compatibility.
- Infrastructure updates: coordinate with `infra/helmfile` maintainers.
