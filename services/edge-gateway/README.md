# Edge Gateway Domain

## Purpose
This modular monolith will unify API, web, and GraphQL gateways into a single, extensible edge service. It will remain polyglot-safe by exposing HTTP/GraphQL contracts defined in `contracts/`.

## Modules (planned)
- `api` – REST endpoints consolidating existing gateway routes.
- `graphql` – GraphQL schema and resolvers derived from consolidated SDL.
- `auth` – shared authentication/authorization middleware and integrations with the identity platform.
- `rate_limit` – standardized rate limiting and caching built from `libs/python/http`.
- `clients` – consumers of downstream domains via generated clients.

## Status
- [x] Directory scaffolded
- [ ] Migrate code from legacy gateways
- [ ] Integrate shared libraries
- [ ] Publish OpenAPI/GraphQL contracts

## Next steps
1. Establish FastAPI/ASGI application entry point consuming shared libs.
2. Import existing auth/rate limiting logic through shared packages.
3. Gradually route traffic through this domain while deprecating legacy services.

