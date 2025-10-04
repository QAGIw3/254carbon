#!/bin/bash
# Security Scanning Script for Production Deployment
#
# Purpose
# - Perform SAST (Semgrep), container image scanning (Trivy), basic network
#   probing (nmap), and Kubernetes posture checks to catch issues before prod.
#
# Usage
#   MIN_SEVERITY=HIGH ./security-scan.sh
#
# Prerequisites
# - kubectl, trivy, semgrep, nmap, jq
# - Cluster context set for target namespace
#
# Outputs
# - JSON scan outputs and Markdown summary under ./security-results

set -euo pipefail

# Configuration
NAMESPACE="market-intelligence"
RESULTS_DIR="./security-results"
MIN_SEVERITY="${MIN_SEVERITY:-HIGH}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi

    if ! command -v trivy &> /dev/null; then
        log_error "Trivy is not installed. Install from: https://trivy.dev"
        exit 1
    fi

    if ! command -v semgrep &> /dev/null; then
        log_error "Semgrep is not installed. Install from: https://semgrep.dev"
        exit 1
    fi

    if ! command -v nmap &> /dev/null; then
        log_error "nmap is not installed"
        exit 1
    fi

    mkdir -p "$RESULTS_DIR"
}

# Scan container images for vulnerabilities
scan_container_images() {
    log_info "Scanning container images for vulnerabilities..."

    # Get all deployment images
    kubectl get deployments -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.spec.template.spec.containers[*].image}{"\n"}{end}' | \
        sort | uniq > "$RESULTS_DIR/images.txt"

    # Scan each image with Trivy
    local vulnerabilities_found=0

    while IFS= read -r image; do
        if [[ -n "$image" && "$image" != "null" ]]; then
            log_info "Scanning image: $image"
            local image_name=$(echo "$image" | cut -d: -f1 | cut -d/ -f2- | tr '/' '-')
            local tag=$(echo "$image" | cut -d: -f2-)

            trivy image \
                --format json \
                --output "$RESULTS_DIR/trivy-$image_name.json" \
                --severity "$MIN_SEVERITY" \
                "$image"

            # Check for vulnerabilities
            local vuln_count=$(jq '.Results[0].Vulnerabilities | length' "$RESULTS_DIR/trivy-$image_name.json" 2>/dev/null || echo "0")
            if [[ "$vuln_count" -gt 0 ]]; then
                log_warn "Found $vuln_count vulnerabilities in $image"
                vulnerabilities_found=$((vulnerabilities_found + vuln_count))
            else
                log_info "No vulnerabilities found in $image"
            fi
        fi
    done < "$RESULTS_DIR/images.txt"

    if [[ "$vulnerabilities_found" -gt 0 ]]; then
        log_warn "Total vulnerabilities found: $vulnerabilities_found"
        return 1
    else
        log_info "No vulnerabilities found in container images"
    fi
}

# Scan source code for security issues
scan_source_code() {
    log_info "Scanning source code for security issues..."

    # Scan Python code with Semgrep
    semgrep \
        --config="p/security-audit" \
        --config="p/owasp-top-ten" \
        --json \
        --output="$RESULTS_DIR/semgrep-python.json" \
        platform/apps/ \
        platform/data/ \
        platform/shared/ || true

    # Scan TypeScript/JavaScript code
    semgrep \
        --config="p/javascript" \
        --config="p/typescript" \
        --json \
        --output="$RESULTS_DIR/semgrep-typescript.json" \
        platform/apps/web-hub/ || true

    # Check for results
    local python_findings=$(jq '.results | length' "$RESULTS_DIR/semgrep-python.json" 2>/dev/null || echo "0")
    local ts_findings=$(jq '.results | length' "$RESULTS_DIR/semgrep-typescript.json" 2>/dev/null || echo "0")

    if [[ "$python_findings" -gt 0 || "$ts_findings" -gt 0 ]]; then
        log_warn "Found $python_findings Python and $ts_findings TypeScript security findings"
        return 1
    else
        log_info "No security issues found in source code"
    fi
}

# Perform network security scan
scan_network_security() {
    log_info "Scanning network security..."

    # Get service IPs
    kubectl get services -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.spec.clusterIP}{" "}{.metadata.name}{"\n"}{end}' | \
        grep -v "None" > "$RESULTS_DIR/services.txt"

    # Scan for open ports (basic check)
    while IFS= read -r line; do
        local ip=$(echo "$line" | awk '{print $1}')
        local service=$(echo "$line" | awk '{print $2}')

        if [[ "$ip" != "None" ]]; then
            log_info "Checking service: $service ($ip)"

            # Use timeout to avoid hanging
            timeout 10s nmap -p 1-65535 "$ip" > "$RESULTS_DIR/nmap-$service.txt" 2>&1 || true

            # Check for unexpected open ports
            local open_ports=$(grep "open" "$RESULTS_DIR/nmap-$service.txt" | wc -l)

            if [[ "$open_ports" -gt 10 ]]; then
                log_warn "Service $service has $open_ports open ports - review required"
                return 1
            else
                log_info "Service $service has acceptable port configuration"
            fi
        fi
    done < "$RESULTS_DIR/services.txt"
}

# Check Kubernetes security posture
check_k8s_security() {
    log_info "Checking Kubernetes security posture..."

    # Check for security contexts
    local pods_without_security_context=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | \
        xargs -I {} kubectl get pod {} -n "$NAMESPACE" -o jsonpath='{.spec.securityContext}' | \
        grep -c "null" || echo "0")

    if [[ "$pods_without_security_context" -gt 0 ]]; then
        log_warn "Found $pods_without_security_context pods without security context"
        return 1
    fi

    # Check for privileged containers
    local privileged_containers=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.securityContext.privileged}{" "}{end}' | \
        grep -c "true" || echo "0")

    if [[ "$privileged_containers" -gt 0 ]]; then
        log_warn "Found $privileged_containers privileged containers"
        return 1
    fi

    # Check for root user containers
    local root_containers=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.securityContext.runAsUser}{" "}{end}' | \
        grep -c "0" || echo "0")

    if [[ "$root_containers" -gt 0 ]]; then
        log_warn "Found $root_containers containers running as root"
        return 1
    fi

    log_info "Kubernetes security posture is good"
}

# Generate security report
generate_report() {
    log_info "Generating security report..."

    cat > "$RESULTS_DIR/security-report.md" << EOF
# Security Scan Report - $(date)

## Summary

$(if [[ -f "$RESULTS_DIR/images.txt" ]]; then echo "- **Images Scanned**: $(wc -l < "$RESULTS_DIR/images.txt")"; fi)
$(if [[ -f "$RESULTS_DIR/semgrep-python.json" ]]; then echo "- **Python Findings**: $(jq '.results | length' "$RESULTS_DIR/semgrep-python.json" 2>/dev/null || echo "0")"; fi)
$(if [[ -f "$RESULTS_DIR/semgrep-typescript.json" ]]; then echo "- **TypeScript Findings**: $(jq '.results | length' "$RESULTS_DIR/semgrep-typescript.json" 2>/dev/null || echo "0")"; fi)

## Recommendations

1. Review and remediate all HIGH and CRITICAL vulnerabilities
2. Implement security contexts on all pods
3. Use non-root users in containers
4. Implement network policies for all services
5. Regular security scanning in CI/CD pipeline

## Next Steps

- Address all findings above $MIN_SEVERITY severity
- Implement automated security scanning in CI/CD
- Schedule regular security assessments
EOF

    log_info "Security report generated: $RESULTS_DIR/security-report.md"
}

# Main execution
main() {
    log_info "Starting comprehensive security scan..."

    check_prerequisites

    local failed_scans=0

    if ! scan_container_images; then
        failed_scans=$((failed_scans + 1))
    fi

    if ! scan_source_code; then
        failed_scans=$((failed_scans + 1))
    fi

    if ! scan_network_security; then
        failed_scans=$((failed_scans + 1))
    fi

    if ! check_k8s_security; then
        failed_scans=$((failed_scans + 1))
    fi

    generate_report

    if [[ "$failed_scans" -gt 0 ]]; then
        log_error "Security scan found $failed_scans issues. Please review and remediate."
        log_info "Report available at: $RESULTS_DIR/security-report.md"
        exit 1
    else
        log_info "All security scans passed! ✓"
        exit 0
    fi
}

# Run main function
main "$@"
