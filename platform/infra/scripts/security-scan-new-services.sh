#!/bin/bash
# Security scan script for new services
# Uses Trivy for vulnerability scanning

set -e

echo "🔒 Security Scanning New Service Docker Images"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0.32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Trivy is installed
if ! command -v trivy &> /dev/null; then
    echo "${RED}Error: Trivy is not installed${NC}"
    echo "Install with: brew install aquasecurity/trivy/trivy"
    echo "Or see: https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
    exit 1
fi

# Services to scan
SERVICES=(
    "marketplace"
    "signals-service"
    "fundamental-models"
    "lmp-decomposition-service"
    "ml-service"
)

# Scan results directory
RESULTS_DIR="security-scan-results"
mkdir -p "$RESULTS_DIR"

# Scan configuration
SEVERITY="CRITICAL,HIGH,MEDIUM"
SCAN_DATE=$(date +%Y%m%d-%H%M%S)

echo "Scan Configuration:"
echo "  Severity levels: $SEVERITY"
echo "  Results directory: $RESULTS_DIR"
echo ""

# Track overall status
TOTAL_CRITICAL=0
TOTAL_HIGH=0
TOTAL_MEDIUM=0

# Scan each service
for SERVICE in "${SERVICES[@]}"; do
    echo "----------------------------------------"
    echo "Scanning: $SERVICE"
    echo "----------------------------------------"
    
    IMAGE_NAME="254carbon/${SERVICE}:latest"
    OUTPUT_FILE="$RESULTS_DIR/${SERVICE}-${SCAN_DATE}.json"
    
    # Run Trivy scan
    echo "Running Trivy scan on ${IMAGE_NAME}..."
    
    trivy image \
        --severity "$SEVERITY" \
        --format json \
        --output "$OUTPUT_FILE" \
        --timeout 10m \
        "$IMAGE_NAME"
    
    # Parse results
    if [ -f "$OUTPUT_FILE" ]; then
        CRITICAL=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL")] | length' "$OUTPUT_FILE")
        HIGH=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH")] | length' "$OUTPUT_FILE")
        MEDIUM=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="MEDIUM")] | length' "$OUTPUT_FILE")
        
        TOTAL_CRITICAL=$((TOTAL_CRITICAL + CRITICAL))
        TOTAL_HIGH=$((TOTAL_HIGH + HIGH))
        TOTAL_MEDIUM=$((TOTAL_MEDIUM + MEDIUM))
        
        echo ""
        echo "Results for $SERVICE:"
        
        if [ "$CRITICAL" -gt 0 ]; then
            echo "${RED}  CRITICAL: $CRITICAL${NC}"
        else
            echo "${GREEN}  CRITICAL: 0${NC}"
        fi
        
        if [ "$HIGH" -gt 0 ]; then
            echo "${YELLOW}  HIGH: $HIGH${NC}"
        else
            echo "${GREEN}  HIGH: 0${NC}"
        fi
        
        echo "  MEDIUM: $MEDIUM"
        echo ""
        echo "Full report saved to: $OUTPUT_FILE"
    else
        echo "${RED}Error: Failed to generate scan report for $SERVICE${NC}"
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo " Security Scan Summary"
echo "=========================================="
echo ""
echo "Total Vulnerabilities Found:"

if [ "$TOTAL_CRITICAL" -gt 0 ]; then
    echo "${RED}  CRITICAL: $TOTAL_CRITICAL${NC}"
else
    echo "${GREEN}  CRITICAL: 0${NC}"
fi

if [ "$TOTAL_HIGH" -gt 0 ]; then
    echo "${YELLOW}  HIGH: $TOTAL_HIGH${NC}"
else
    echo "${GREEN}  HIGH: 0${NC}"
fi

echo "  MEDIUM: $TOTAL_MEDIUM"
echo ""

# Exit status
if [ "$TOTAL_CRITICAL" -gt 0 ]; then
    echo "${RED}❌ CRITICAL vulnerabilities found - remediation required${NC}"
    exit 1
elif [ "$TOTAL_HIGH" -gt 0 ]; then
    echo "${YELLOW}⚠️  HIGH severity vulnerabilities found - review recommended${NC}"
    exit 0
else
    echo "${GREEN}✅ No critical or high severity vulnerabilities found${NC}"
    exit 0
fi

