#!/bin/bash

# 254Carbon Platform - Database Reset Script
# WARNING: This will delete all data and reinitialize the databases

set -e

echo "⚠️  WARNING: This will delete ALL data in PostgreSQL and ClickHouse!"
echo ""
echo "This action cannot be undone. Are you sure you want to continue?"
read -p "Type 'YES' to confirm: " confirm

if [ "$confirm" != "YES" ]; then
    echo "❌ Operation cancelled."
    exit 0
fi

echo "🗑️  Resetting databases..."

# Navigate to platform directory
cd "$(dirname "$0")/.."

echo "📦 Stopping all services..."
docker-compose down

echo "🗃️  Cleaning data volumes..."

# Remove and recreate data volumes
docker volume rm 254carbon_postgres_data 254carbon_clickhouse_data 254carbon_kafka_data 254carbon_minio_data 2>/dev/null || true

echo "🚀 Starting infrastructure services..."
docker-compose up -d postgres clickhouse kafka minio

echo "⏳ Waiting for services to be ready..."

# Wait for PostgreSQL
timeout=60
while [ $timeout -gt 0 ]; do
    if docker-compose exec -T postgres pg_isready -U postgres -d market_intelligence > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo "❌ PostgreSQL failed to start"
    exit 1
fi

# Wait for ClickHouse
timeout=60
while [ $timeout -gt 0 ]; do
    if curl -s http://localhost:8123/ > /dev/null 2>&1; then
        echo "✅ ClickHouse is ready"
        break
    fi
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo "❌ ClickHouse failed to start"
    exit 1
fi

echo "🗃️  Reinitializing schemas..."

# Reinitialize PostgreSQL schema
echo "Reinitializing PostgreSQL schema..."
docker-compose exec -T postgres psql -U postgres -d market_intelligence -f /docker-entrypoint-initdb.d/init.sql > /dev/null 2>&1

# Reinitialize ClickHouse schema
echo "Reinitializing ClickHouse schema..."
curl -s -X POST 'http://localhost:8123/' \
     --header 'Content-Type: application/octet-stream' \
     --data-binary @./data/schemas/clickhouse/init.sql > /dev/null 2>&1

echo "🌱 Seeding fresh sample data..."
./scripts/seed-data.sh

echo "🔧 Restarting application services..."
docker-compose up -d gateway curve-service scenario-engine report-service backtesting-service download-center web-hub

echo "✅ Database reset complete!"
echo ""
echo "📋 Fresh Environment Ready:"
echo "   • All data volumes recreated"
echo "   • Schemas reinitialized"
echo "   • Sample data seeded"
echo "   • All services restarted"
echo ""
echo "🎯 Next steps:"
echo "   • Test API endpoints: http://localhost:8000/docs"
echo "   • Access Web Hub: http://localhost:3000"
echo "   • Generate reports and run scenarios"
echo ""
