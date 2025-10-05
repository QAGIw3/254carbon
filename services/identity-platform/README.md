# Identity Platform Domain

## Purpose
The identity platform consolidates authentication, authorization, and entitlements into a single cohesive service with clear interfaces to the edge gateway and other domains.

## Modules (planned)
- `auth` – token issuance, JWKS publishing, and session handling.
- `entitlements` – role and attribute-based access control engines.
- `audit` – security event logging and policy enforcement integration.
- `api` – public APIs for identity, permissions, and introspection.
- `clients` – shared client SDKs exposed via `libs/python/clients` and `libs/js/clients`.

## Status
- [x] Directory scaffolded
- [ ] Migrate auth-service and entitlements-service codebases
- [ ] Expose unified OpenAPI contract
- [ ] Integrate with shared logging and metrics

## Migration considerations
- Maintain token formats and JWKS URLs to avoid breaking existing consumers.
- Provide migration scripts for database schema changes via Alembic.
- Ensure idempotent operations for role assignments and entitlements updates.

## Migration Status

- [x] Directory scaffolded
- [ ] Relocate authentication logic from `platform/apps/auth-service`
- [ ] Relocate entitlements engine from `platform/apps/entitlements-service`
- [ ] Integrate shared DB session management and audit logging
- [ ] Expose unified OpenAPI contract from `contracts/http/identity-platform`

## Dependencies

- Requires `carbon254-libs-python` for config, db, http, and logging modules
- Depends on Keycloak/IdP configurations managed via Helm/Vault
- Publishes audit events defined in `contracts/events/identity`

