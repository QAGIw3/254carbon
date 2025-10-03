#!/bin/bash
# Initialize database schemas

set -e

echo "=== Initializing Database Schemas ==="

# Wait for services to be ready
echo "Waiting for PostgreSQL..."
until kubectl exec -n market-intelligence-infra postgresql-0 -- pg_isready > /dev/null 2>&1; do
  sleep 2
done

echo "Waiting for ClickHouse..."
until kubectl exec -n market-intelligence-infra clickhouse-0 -- clickhouse-client --query "SELECT 1" > /dev/null 2>&1; do
  sleep 2
done

# Initialize PostgreSQL schema
echo "Initializing PostgreSQL schema..."
kubectl exec -n market-intelligence-infra postgresql-0 -- psql -U postgres -d market_intelligence < postgres/init.sql

# Initialize ClickHouse schema  
echo "Initializing ClickHouse schema..."
kubectl exec -n market-intelligence-infra clickhouse-0 -- clickhouse-client --multiquery < clickhouse/init.sql

echo "=== Database initialization complete ==="

