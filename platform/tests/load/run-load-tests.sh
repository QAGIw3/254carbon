#!/bin/bash
# Run comprehensive load tests against 254Carbon API
# Validates SLA compliance before production deployment

set -euo pipefail

# Configuration
API_URL="${API_URL:-https://api.254carbon.ai}"
WS_URL="${WS_URL:-wss://api.254carbon.ai/ws/stream}"
AUTH_TOKEN="${AUTH_TOKEN:-}"
RESULTS_DIR="./results"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v k6 &> /dev/null; then
        log_error "k6 is not installed. Install from: https://k6.io/docs/get-started/installation/"
        exit 1
    fi
    
    if [ -z "$AUTH_TOKEN" ]; then
        log_warn "AUTH_TOKEN not set. Using default test token."
        AUTH_TOKEN="test-token"
    fi
    
    mkdir -p "$RESULTS_DIR"
}

# Test API health before load testing
test_api_health() {
    log_info "Testing API health..."
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
    
    if [ "$response" != "200" ]; then
        log_error "API health check failed (HTTP $response)"
        exit 1
    fi
    
    log_info "API is healthy"
}

# Run API load test
run_api_load_test() {
    log_info "Running API load test..."
    
    k6 run \
        --out json="$RESULTS_DIR/api-load-test.json" \
        --out influxdb=http://localhost:8086/k6 \
        -e API_URL="$API_URL" \
        -e AUTH_TOKEN="$AUTH_TOKEN" \
        api-load-test.js
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "API load test PASSED ✓"
    else
        log_error "API load test FAILED ✗"
        return 1
    fi
}

# Run streaming load test
run_streaming_load_test() {
    log_info "Running WebSocket streaming load test..."
    
    k6 run \
        --out json="$RESULTS_DIR/streaming-load-test.json" \
        -e WS_URL="$WS_URL" \
        -e AUTH_TOKEN="$AUTH_TOKEN" \
        streaming-load-test.js
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "Streaming load test PASSED ✓"
    else
        log_error "Streaming load test FAILED ✗"
        return 1
    fi
}

# Run soak test (extended duration)
run_soak_test() {
    log_info "Running soak test (30 minutes)..."
    log_warn "This will take ~30 minutes. Press Ctrl+C to skip."
    
    k6 run \
        --duration 30m \
        --vus 100 \
        --out json="$RESULTS_DIR/soak-test.json" \
        -e API_URL="$API_URL" \
        -e AUTH_TOKEN="$AUTH_TOKEN" \
        api-load-test.js
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "Soak test PASSED ✓"
    else
        log_error "Soak test FAILED ✗"
        return 1
    fi
}

# Analyze results
analyze_results() {
    log_info "Analyzing test results..."
    
    if [ -f "$RESULTS_DIR/api-load-test.json" ]; then
        # Extract key metrics using jq
        local p95_latency=$(jq -r '.metrics.http_req_duration.values["p(95)"]' "$RESULTS_DIR/api-load-test.json")
        local error_rate=$(jq -r '.metrics.errors.values.rate' "$RESULTS_DIR/api-load-test.json")
        local total_requests=$(jq -r '.metrics.requests.values.count' "$RESULTS_DIR/api-load-test.json")
        
        echo ""
        echo "══════════════════════════════════════"
        echo "  Load Test Summary"
        echo "══════════════════════════════════════"
        echo "  Total Requests: $total_requests"
        echo "  p95 Latency: ${p95_latency}ms (SLA: <250ms)"
        echo "  Error Rate: $(echo "$error_rate * 100" | bc)% (SLA: <1%)"
        echo "══════════════════════════════════════"
        echo ""
        
        # Check SLA compliance
        if (( $(echo "$p95_latency < 250" | bc -l) )); then
            log_info "✓ Latency SLA met"
        else
            log_error "✗ Latency SLA violated"
        fi
        
        if (( $(echo "$error_rate < 0.01" | bc -l) )); then
            log_info "✓ Error rate SLA met"
        else
            log_error "✗ Error rate SLA violated"
        fi
    fi
}

# Generate HTML report
generate_report() {
    log_info "Generating HTML report..."
    
    if command -v k6-html-reporter &> /dev/null; then
        k6-html-reporter "$RESULTS_DIR/api-load-test.json" \
            --output "$RESULTS_DIR/load-test-report.html"
        
        log_info "Report generated: $RESULTS_DIR/load-test-report.html"
    else
        log_warn "k6-html-reporter not installed. Skipping HTML report."
        log_info "Install with: npm install -g k6-html-reporter"
    fi
}

# Main execution
main() {
    log_info "Starting 254Carbon load tests..."
    log_info "API URL: $API_URL"
    log_info "Results directory: $RESULTS_DIR"
    echo ""
    
    check_prerequisites
    test_api_health
    
    # Run tests
    run_api_load_test || exit 1
    
    log_info "Waiting 30 seconds before streaming test..."
    sleep 30
    
    run_streaming_load_test || exit 1
    
    # Optional: soak test
    read -p "Run soak test (30 minutes)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_soak_test
    fi
    
    # Analyze and report
    analyze_results
    generate_report
    
    log_info "Load testing completed!"
    log_info "Results saved to: $RESULTS_DIR"
}

# Run main
main "$@"



