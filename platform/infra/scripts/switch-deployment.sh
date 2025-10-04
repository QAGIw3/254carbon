#!/bin/bash

# 254Carbon Blue-Green Deployment Switcher
#
# Purpose
# - Safely switch service traffic between blue and green Kubernetes deployments.
#
# Usage
#   ./switch-deployment.sh <namespace> <service_name> <blue|green>
#   e.g., ./switch-deployment.sh market-intelligence api-gateway blue
#
# Behavior
# - Validates inputs and current context, scales up target color, waits ready,
#   flips service selector, then scales down previous color. Rolls back on
#   failure if health checks do not pass.

set -euo pipefail

# Configuration
NAMESPACE=${1:-market-intelligence}
SERVICE_NAME=${2:-market-intelligence}
NEW_COLOR=${3:-}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# Check prerequisites
check_prerequisites() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not installed"
        exit 1
    fi

    if ! kubectl config current-context &> /dev/null; then
        log_error "kubectl not configured"
        exit 1
    fi
}

# Get current active color
get_current_color() {
    local current_replicas
    current_replicas=$(kubectl get deployment -n "$NAMESPACE" -l "app.kubernetes.io/instance=${SERVICE_NAME},app.kubernetes.io/color" -o jsonpath='{.items[?(@.spec.replicas>0)].metadata.labels.app\.kubernetes\.io/color}' | tr ' ' '\n' | head -1)

    if [ -z "$current_replicas" ]; then
        log_error "No active deployment found"
        exit 1
    fi

    echo "$current_replicas"
}

# Validate new color
validate_color() {
    local color="$1"

    if [[ "$color" != "blue" && "$color" != "green" ]]; then
        log_error "Invalid color: $color. Must be 'blue' or 'green'"
        exit 1
    fi
}

# Wait for deployment to be ready
wait_for_deployment() {
    local deployment_name="$1"
    local timeout="${2:-300}"

    log_info "Waiting for deployment $deployment_name to be ready..."

    kubectl wait --for=condition=available --timeout="${timeout}s" deployment/"$deployment_name" -n "$NAMESPACE"

    # Additional check for all pods to be ready
    kubectl wait --for=condition=ready --timeout="${timeout}s" pod -l "app.kubernetes.io/instance=${deployment_name}" -n "$NAMESPACE"
}

# Scale down old deployment
scale_down_deployment() {
    local deployment_name="$1"
    local current_color="$2"

    log_info "Scaling down $deployment_name deployment..."

    kubectl scale deployment "$deployment_name" --replicas=0 -n "$NAMESPACE"

    # Wait for pods to terminate
    kubectl wait --for=delete --timeout=300s pod -l "app.kubernetes.io/instance=${deployment_name}" -n "$NAMESPACE" || true
}

# Scale up new deployment
scale_up_deployment() {
    local deployment_name="$1"
    local desired_replicas="$2"

    log_info "Scaling up $deployment_name deployment to $desired_replicas replicas..."

    kubectl scale deployment "$deployment_name" --replicas="$desired_replicas" -n "$NAMESPACE"

    wait_for_deployment "$deployment_name" 300
}

# Update service selector
update_service_selector() {
    local new_color="$1"

    log_info "Updating service selector to point to $new_color deployment..."

    # Patch the service to select the new color
    kubectl patch service "$SERVICE_NAME" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app.kubernetes.io/color\":\"$new_color\"}}}"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."

    # Check if service is responding
    local service_url
    service_url=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')

    if [ -n "$service_url" ]; then
        # Try to access health endpoint (adjust port as needed)
        kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -n "$NAMESPACE" -- curl -f "http://$service_url:8000/health" || {
            log_error "Health check failed"
            return 1
        }
    fi

    log_success "Health checks passed"
}

# Rollback function
rollback() {
    local original_color="$1"
    local new_color="$2"

    log_warn "Rolling back to $original_color deployment..."

    scale_up_deployment "${SERVICE_NAME}-${original_color}" 3
    update_service_selector "$original_color"
    scale_down_deployment "${SERVICE_NAME}-${new_color}" "$new_color"

    log_success "Rollback completed"
}

# Main deployment switch function
switch_deployment() {
    local current_color
    current_color=$(get_current_color)

    if [ -z "$NEW_COLOR" ]; then
        if [ "$current_color" = "blue" ]; then
            NEW_COLOR="green"
        else
            NEW_COLOR="blue"
        fi
    fi

    validate_color "$NEW_COLOR"

    if [ "$current_color" = "$NEW_COLOR" ]; then
        log_info "Already on $NEW_COLOR deployment"
        exit 0
    fi

    local target_deployment="${SERVICE_NAME}-${NEW_COLOR}"

    log_info "Switching from $current_color to $NEW_COLOR deployment..."

    # Get desired replicas for the target deployment
    local desired_replicas
    desired_replicas=$(kubectl get deployment -n "$NAMESPACE" -l "app.kubernetes.io/color=$NEW_COLOR" -o jsonpath='{.items[0].spec.replicas}')

    # Scale up new deployment
    scale_up_deployment "$target_deployment" "$desired_replicas"

    # Update service selector
    update_service_selector "$NEW_COLOR"

    # Wait a bit for traffic to settle
    sleep 30

    # Run health checks
    if ! run_health_checks; then
        log_error "Health checks failed, rolling back..."
        rollback "$current_color" "$NEW_COLOR"
        exit 1
    fi

    # Scale down old deployment
    scale_down_deployment "${SERVICE_NAME}-${current_color}" "$current_color"

    log_success "Successfully switched to $NEW_COLOR deployment!"
}

# Main execution
main() {
    check_prerequisites

    if [ $# -lt 1 ]; then
        echo "Usage: $0 <namespace> [service-name] [new-color]"
        echo "Example: $0 market-intelligence market-intelligence green"
        exit 1
    fi

    switch_deployment
}

main "$@"
