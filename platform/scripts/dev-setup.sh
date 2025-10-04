#!/bin/bash

# 254Carbon Platform - Development Environment Setup
# One-command setup for local development environment
# - Starts core infra (Postgres/ClickHouse/Kafka/MinIO/Keycloak)
# - Waits for health and performs basic readiness checks

set -e

echo "🚀 Setting up 254Carbon development environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Navigate to platform directory
cd "$(dirname "$0")/.."

echo "📦 Starting infrastructure services..."

# Start core infrastructure services
docker-compose up -d postgres clickhouse kafka minio keycloak

echo "⏳ Waiting for services to be healthy..."

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
timeout=60
while [ $timeout -gt 0 ]; do
    if docker-compose exec -T postgres pg_isready -U postgres -d market_intelligence > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    echo "⏳ Still waiting... (${timeout}s)"
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo "❌ PostgreSQL failed to start within timeout"
    exit 1
fi

# Wait for ClickHouse to be ready
echo "Waiting for ClickHouse..."
timeout=60
while [ $timeout -gt 0 ]; do
    if curl -s http://localhost:8123/ > /dev/null 2>&1; then
        echo "✅ ClickHouse is ready"
        break
    fi
    echo "⏳ Still waiting... (${timeout}s)"
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo "❌ ClickHouse failed to start within timeout"
    exit 1
fi

# Wait for Kafka to be ready
echo "Waiting for Kafka..."
timeout=60
while [ $timeout -gt 0 ]; do
    if docker-compose exec -T kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
        echo "✅ Kafka is ready"
        break
    fi
    echo "⏳ Still waiting... (${timeout}s)"
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo "❌ Kafka failed to start within timeout"
    exit 1
fi

echo "🗃️  Initializing databases..."

# Initialize PostgreSQL schema
echo "Initializing PostgreSQL schema..."
docker-compose exec -T postgres psql -U postgres -d market_intelligence -f /docker-entrypoint-initdb.d/init.sql > /dev/null 2>&1

# Initialize ClickHouse schema
echo "Initializing ClickHouse schema..."
curl -s -X POST 'http://localhost:8123/' \
     --header 'Content-Type: application/octet-stream' \
     --data-binary @./data/schemas/clickhouse/init.sql > /dev/null 2>&1

echo "🌱 Seeding sample data..."

# Run sample data seeding script if it exists
if [ -f "./scripts/seed-data.sh" ]; then
    echo "Seeding sample data..."
    ./scripts/seed-data.sh
fi

echo "🔧 Starting application services..."

# Start application services
docker-compose up -d gateway curve-service scenario-engine report-service backtesting-service download-center web-hub

echo "⏳ Waiting for application services..."

# Wait for gateway to be ready
echo "Waiting for API Gateway..."
timeout=120
while [ $timeout -gt 0 ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API Gateway is ready"
        break
    fi
    echo "⏳ Still waiting... (${timeout}s)"
    sleep 3
    timeout=$((timeout - 3))
done

if [ $timeout -le 0 ]; then
    echo "❌ API Gateway failed to start within timeout"
    exit 1
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Access Points:"
echo "   🌐 Web Hub:        http://localhost:3000"
echo "   🔌 API Gateway:    http://localhost:8000"
echo "   📚 API Docs:       http://localhost:8000/docs"
echo "   🔑 Keycloak:       http://localhost:8080"
echo "   📊 Grafana:        http://localhost:3001"
echo ""
echo "🛠️  Development Commands:"
echo "   Start all:         docker-compose up"
echo "   Stop all:          docker-compose down"
echo "   View logs:         docker-compose logs -f [service-name]"
echo "   Reset database:    ./scripts/reset-db.sh"
echo ""
echo "💡 Tips:"
echo "   - All services support hot-reloading"
echo "   - Use Ctrl+C to stop services gracefully"
echo "   - Check logs with: docker-compose logs -f"
echo ""
echo "🚀 Happy coding!"
