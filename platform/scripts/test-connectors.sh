#!/bin/bash
# Smoke test selected connectors in local/dev mode

# 254Carbon Platform - Connector Testing Script
# Validates that data connectors are working correctly

set -e

echo "🔌 Testing data connectors..."

# Function to test a connector service
test_connector() {
    local service_name=$1
    local port=$2
    local expected_status="healthy"

    echo "Testing $service_name..."

    # Check if service is running
    if ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "❌ $service_name is not running or not accessible"
        return 1
    fi

    # Get health status
    health_response=$(curl -s "http://localhost:$port/health")

    if echo "$health_response" | grep -q '"status": "healthy"'; then
        echo "✅ $service_name is healthy"
    else
        echo "❌ $service_name health check failed: $health_response"
        return 1
    fi

    # Test discovery endpoint if available
    if curl -s "http://localhost:$port/api/v1/discover" > /dev/null 2>&1; then
        echo "✅ $service_name discovery endpoint accessible"
    fi

    return 0
}

# Function to test data availability
test_data_availability() {
    echo "Testing data availability..."

    # Test ClickHouse data
    echo "Testing ClickHouse data access..."
    if curl -s -X POST 'http://localhost:8123/' \
         --header 'Content-Type: application/octet-stream' \
         --data-binary "SELECT COUNT(*) FROM market_price_ticks" \
         | grep -q '[0-9]\+'; then
        echo "✅ ClickHouse data accessible"
    else
        echo "❌ ClickHouse data not accessible"
        return 1
    fi

    # Test PostgreSQL data
    echo "Testing PostgreSQL data access..."
    if docker-compose exec -T postgres psql -U postgres -d market_intelligence -c "SELECT COUNT(*) FROM pg.instrument;" \
         | grep -q '[0-9]\+'; then
        echo "✅ PostgreSQL data accessible"
    else
        echo "❌ PostgreSQL data not accessible"
        return 1
    fi

    return 0
}

# Test individual connector services
echo "🧪 Testing connector services..."

failed_connectors=()
services_to_test=(
    "gateway:8000"
    "curve-service:8001"
    "scenario-engine:8002"
    "report-service:8004"
    "backtesting-service:8005"
    "download-center:8006"
)

for service_info in "${services_to_test[@]}"; do
    service_name=$(echo "$service_info" | cut -d: -f1)
    port=$(echo "$service_info" | cut -d: -f2)

    if ! test_connector "$service_name" "$port"; then
        failed_connectors+=("$service_name")
    fi
done

# Test data availability
if ! test_data_availability; then
    echo "❌ Data availability test failed"
    exit 1
fi

# Summary
echo ""
echo "📊 Test Summary:"

if [ ${#failed_connectors[@]} -eq 0 ]; then
    echo "✅ All connector tests passed!"
    echo ""
    echo "🎯 Connectors are ready for:"
    echo "   • Data ingestion and processing"
    echo "   • Forward curve generation"
    echo "   • Scenario modeling"
    echo "   • Report generation"
    echo "   • Backtesting"
else
    echo "❌ Some connector tests failed:"
    for failed in "${failed_connectors[@]}"; do
        echo "   • $failed"
    done
    echo ""
    echo "💡 Troubleshooting:"
    echo "   • Check service logs: docker-compose logs [service-name]"
    echo "   • Verify database connections"
    echo "   • Check network connectivity"
    exit 1
fi

echo ""
echo "🚀 All systems operational!"
