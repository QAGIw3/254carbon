LMP Components Migration (ClickHouse + Sink Mapping)

Scope
- Promote LMP subcomponents to first-class columns in ClickHouse and include in ingestion mapping:
  - energy_component
  - congestion_component
  - loss_component

Prereqs
- ClickHouse reachable with DDL privileges
- Kafka → ClickHouse ingestion (Kafka Engine + MV or external consumer) under your control

DDL (safe, idempotent)
Run on the cluster (adjust database if not `market_intelligence`):

```
ALTER TABLE market_intelligence.market_price_ticks
    ADD COLUMN IF NOT EXISTS energy_component Nullable(Decimal(10,4)) CODEC(T64, LZ4) AFTER value;

ALTER TABLE market_intelligence.market_price_ticks
    ADD COLUMN IF NOT EXISTS congestion_component Nullable(Decimal(10,4)) CODEC(T64, LZ4) AFTER energy_component;

ALTER TABLE market_intelligence.market_price_ticks
    ADD COLUMN IF NOT EXISTS loss_component Nullable(Decimal(10,4)) CODEC(T64, LZ4) AFTER congestion_component;
```

Sink Mapping
- If using ClickHouse Kafka Engine + Materialized View:
  - Add fields to source Kafka table schema and MV select list.
  - Example mapping included in: `platform/data/schemas/clickhouse/migrations/2025_10_add_lmp_components.sql` (commented).

- If using an external consumer/ETL:
  - Map JSON keys → columns:
    - `energy_component` → `market_intelligence.market_price_ticks.energy_component`
    - `congestion_component` → `...congestion_component`
    - `loss_component` → `...loss_component`
  - Coerce types to Decimal(10,4); allow nulls.

Zero‑Downtime Plan
1) Apply ALTERs (no lock on MergeTree structure)
2) Deploy ingestion updates:
   - Connectors now publish the new keys (already in repo for CAISO/PJM/SPP/NYISO/MISO where available)
   - Update your consumer/MV mapping to include the new fields
3) Validate:
   - Ingest a small window (e.g., CAISO RTM hubs)
   - Query:
     - `SELECT count(), sum(isNotNull(energy_component)) FROM market_intelligence.market_price_ticks WHERE source LIKE 'caiso_%' AND event_time >= now() - INTERVAL 1 HOUR;`
   - Check distribution/ranges for components
4) Backfill (optional):
   - Use `caiso_backfill` DAG for recent windows if you want to populate components historically

Roll‑back
- The new columns are additive. If ingestion fails, you can revert consumer changes and leave columns unused

FAQ
- Are components always present?
  - No. Some ISOs do not expose components in all feeds; values will be NULL.
- Any index changes needed?
  - Not required; consider sparse indexes if heavily queried by components.

