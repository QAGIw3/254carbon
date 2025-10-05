# Commodities Service

Unified service for commodity market data across gas, oil, coal, biofuels, and battery materials.

## Overview

- Framework: FastAPI (ASGI)
- Port: `8012`
- Routers: `gas`, `oil`, `coal`, `biofuels`, `battery-materials`
- Data Source: ClickHouse (primary), synthetic fallbacks for dev parity
- Cache: Redis via shared `CacheManager` with TTL tiers
- Auth: Minimal JWT verification + role checks with optional `LOCAL_DEV` bypass

## Endpoints (MVP)

- `GET /api/v1/commodities/gas/prices` – Henry Hub/basis prices
- `GET /api/v1/commodities/gas/storage` – EIA-like weekly storage
- `GET /api/v1/commodities/gas/pipelines` – Pipeline flows/utilization
- `GET /api/v1/commodities/gas/lng` – LNG feedgas snapshots
- `GET /api/v1/commodities/oil/curves` – Futures curves (WTI/Brent)
- `GET /api/v1/commodities/coal/indices` – Index prices (API2/Newcastle)
- `GET /api/v1/commodities/coal/stockpiles` – Stockpile estimates
- `GET /api/v1/commodities/biofuels/rin-prices` – RIN prices
- `GET /api/v1/commodities/battery-materials/lithium` – Lithium prices + supply chain snapshot

## Environment Variables

- `CLICKHOUSE_HOST`/`CLICKHOUSE_PORT`: ClickHouse connection (default `clickhouse:9000`)
- `DATABASE_URL`: Postgres DSN for optional metadata/entitlement (default dev DSN)
- `REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`: Redis connection for cache (defaults `redis:6379/0`)
- `CACHE_PREFIX`/`CACHE_DEFAULT_TTL`: Cache namespacing and TTL tuning
- `LOCAL_DEV`: `true` to bypass JWT and grant common roles locally
- `KEYCLOAK_URL`/`KEYCLOAK_AUDIENCE`: Optional JWT verification settings

## Caching Strategy

- REALTIME (≈45s): fast-moving data (hub prices, emissions)
- SEMI_STATIC (≈5–15m): curves and analytics
- STATIC (≈1–6h): metadata/specs

Cache keys are derived from request params via a normalized, hashed payload.

## Fallbacks

If ClickHouse is unreachable or returns no rows, routers generate deterministic, realistic series to keep the API usable for local development and demos.

## Health and Readiness

- `GET /health`: liveness check
- `GET /ready`: readiness (pings ClickHouse, Postgres, and Redis)

## Local Development

Using docker-compose (from repo root):

```
docker compose -f platform/docker-compose.yml up -d clickhouse postgres redis gateway commodities-service
```

Service will be available on `http://localhost:8012`. Gateway proxies `/api/v1/commodities/*` to this service.

## Testing

```
pytest platform/apps/commodities-service/tests/test_routes.py
```

## Directory Structure

- `main.py` – application setup and router includes
- `deps/` – shared helpers (cache, db, auth, query builders)
- `schemas/` – shared pydantic models (prices, curves)
- `gas|oil|coal|biofuels|battery_materials/` – commodity routers/models/synthetic
- `tests/` – smoke tests under LOCAL_DEV bypass

