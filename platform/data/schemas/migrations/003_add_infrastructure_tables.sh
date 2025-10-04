#!/bin/bash
# Migration: Add infrastructure data tables and schemas
# Version: 003
# Date: 2024-01-10

set -e

echo "Running infrastructure schema migration..."

# PostgreSQL migrations
echo "Applying PostgreSQL infrastructure schema..."
PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql \
    -h "${POSTGRES_HOST:-postgresql}" \
    -p "${POSTGRES_PORT:-5432}" \
    -U "${POSTGRES_USER:-postgres}" \
    -d "${POSTGRES_DB:-market_intelligence}" \
    -f /schemas/postgres/infrastructure_schema.sql

echo "PostgreSQL infrastructure schema applied successfully."

# ClickHouse migrations
echo "Applying ClickHouse infrastructure schema..."
clickhouse-client \
    --host "${CLICKHOUSE_HOST:-clickhouse}" \
    --port "${CLICKHOUSE_PORT:-9000}" \
    --user "${CLICKHOUSE_USER:-default}" \
    --password "${CLICKHOUSE_PASSWORD:-}" \
    --multiquery \
    --queries-file /schemas/clickhouse/infrastructure_schema.sql

echo "ClickHouse infrastructure schema applied successfully."

# Register infrastructure data sources
echo "Registering infrastructure data sources..."
PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql \
    -h "${POSTGRES_HOST:-postgresql}" \
    -p "${POSTGRES_PORT:-5432}" \
    -U "${POSTGRES_USER:-postgres}" \
    -d "${POSTGRES_DB:-market_intelligence}" \
    <<EOF
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json, metric_types, fallback_role)
VALUES 
    ('alsi_lng_inventory', 'vendor', 'https://alsi.gie.eu/api', 'active', 
     '{"granularity": "terminal", "lookback_days": 30}', 
     ARRAY['lng_inventory_gwh', 'lng_send_out_gwh', 'lng_ship_arrivals'], 
     'primary'),
    
    ('reexplorer_renewable', 'vendor', 'https://developer.nrel.gov/api/reexplorer/v1', 'active',
     '{"resource_types": ["solar_ghi", "wind_speed_100m"], "include_projects": true}',
     ARRAY['solar_resource', 'wind_resource', 'renewable_capacity'],
     'primary'),
    
    ('wri_powerplants', 'vendor', 'https://datasets.wri.org/dataset/globalpowerplantdatabase', 'active',
     '{"min_capacity_mw": 10, "include_generation": true}',
     ARRAY['power_plant_capacity', 'power_plant_generation'],
     'primary'),
    
    ('gem_transmission', 'vendor', 'https://api.globalenergymonitor.org/v1', 'active',
     '{"min_voltage_kv": 100, "include_projects": true}',
     ARRAY['transmission_capacity', 'transmission_flows', 'infrastructure_projects'],
     'primary')
ON CONFLICT (source_id) DO UPDATE SET
    status = EXCLUDED.status,
    cfg_json = EXCLUDED.cfg_json,
    metric_types = EXCLUDED.metric_types,
    fallback_role = EXCLUDED.fallback_role;
EOF

echo "Infrastructure data sources registered."

# Create Kafka topics for infrastructure data
echo "Creating Kafka topics for infrastructure data..."
kafka-topics.sh --create --if-not-exists \
    --bootstrap-server "${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}" \
    --topic market.infrastructure \
    --partitions 12 \
    --replication-factor 1 \
    --config retention.ms=604800000

echo "Migration 003 completed successfully."
