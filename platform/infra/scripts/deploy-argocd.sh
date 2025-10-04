#!/bin/bash

# 254Carbon ArgoCD GitOps Deployment Script
#
# Purpose
# - Install and bootstrap ArgoCD for GitOps‑driven application delivery.
#
# Usage
#   ./deploy-argocd.sh [namespace] [environment]
#   e.g., ./deploy-argocd.sh argocd dev
#
# Prerequisites
# - kubectl configured to the target cluster/context
# - helm installed and authenticated as needed
# - Optional: argocd CLI installed for initial admin login steps
#
# Notes
# - Admin password is sourced from the initial secret; rotate post‑install.
# - The server service is exposed via LoadBalancer by default; adjust for your
#   environment (Ingress, NodePort) as needed.

set -euo pipefail

# Configuration
NAMESPACE=${1:-argocd}
ENVIRONMENT=${2:-dev}

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

    # Check if ArgoCD CLI is installed
    if ! command -v argocd &> /dev/null; then
        log_warn "ArgoCD CLI not installed, will use kubectl instead"
        ARGOCD_CLI_AVAILABLE=false
    else
        ARGOCD_CLI_AVAILABLE=true
    fi
}

# Install ArgoCD
install_argocd() {
    log_info "Installing ArgoCD..."

    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Install ArgoCD using Helm
    helm repo add argo https://argoproj.github.io/argo-helm
    helm repo update

    helm upgrade --install argocd argo/argo-cd \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --values - <<EOF
server:
  service:
    type: LoadBalancer
  config:
    repositories: |
      - type: git
        name: 254carbon
        url: https://github.com/254carbon/market-intelligence

configs:
  secret:
    argocdServerAdminPassword: "$2a$10\$somehashedpassword"

dex:
  enabled: false

redis:
  enabled: true

controller:
  args:
    app-resync-period: "180"
    repo-server-timeout-seconds: "60"
EOF

    log_success "ArgoCD installed"
}

# Wait for ArgoCD to be ready
wait_for_argocd() {
    log_info "Waiting for ArgoCD to be ready..."

    kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/argocd-repo-server -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/argocd-application-controller -n "$NAMESPACE"

    # Get the ArgoCD server password
    if [ "$ARGOCD_CLI_AVAILABLE" = true ]; then
        kubectl port-forward svc/argocd-server -n "$NAMESPACE" 8080:443 &
        PORT_FORWARD_PID=$!
        sleep 5

        # Login to ArgoCD
        argocd login localhost:8080 --username admin --password "$(kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath='{.data.password}' | base64 -d)" --insecure

        # Kill port forward
        kill $PORT_FORWARD_PID
    fi

    log_success "ArgoCD is ready"
}

# Deploy ArgoCD applications
deploy_applications() {
    log_info "Deploying ArgoCD applications..."

    # Apply project first
    kubectl apply -f ../argocd/projects/market-intelligence-project.yaml

    # Apply applications based on environment
    case "$ENVIRONMENT" in
        "dev")
            kubectl apply -f ../argocd/applications/market-intelligence-dev.yaml
            ;;
        "staging")
            kubectl apply -f ../argocd/applications/market-intelligence-staging.yaml
            ;;
        "prod")
            kubectl apply -f ../argocd/applications/market-intelligence-prod.yaml
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac

    log_success "ArgoCD applications deployed"
}

# Configure webhooks for automated deployment
configure_webhooks() {
    log_info "Configuring webhooks for automated deployment..."

    # Create webhook secret for GitHub
    kubectl create secret generic github-webhook-secret \
        --namespace="$NAMESPACE" \
        --from-literal=webhook-secret="$(openssl rand -hex 20)" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_info "Webhook secret created. Update your GitHub repository webhook with:"
    echo "  URL: https://your-domain/api/webhook"
    echo "  Secret: $(kubectl get secret github-webhook-secret -n $NAMESPACE -o jsonpath='{.data.webhook-secret}' | base64 -d)"

    log_success "Webhooks configured"
}

# Create monitoring for ArgoCD
setup_monitoring() {
    log_info "Setting up monitoring for ArgoCD..."

    # Apply ArgoCD metrics service monitor
    kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: argocd-metrics
  namespace: $NAMESPACE
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: argocd-metrics
  endpoints:
  - port: metrics
    interval: 30s
EOF

    log_success "ArgoCD monitoring configured"
}

# Main deployment flow
main() {
    log_info "Starting ArgoCD GitOps deployment for $ENVIRONMENT environment"

    check_prerequisites
    install_argocd
    wait_for_argocd
    deploy_applications
    configure_webhooks
    setup_monitoring

    log_success "🎉 ArgoCD GitOps deployment completed successfully!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Access ArgoCD UI at: https://your-domain/argocd"
    log_info "2. Update GitHub repository webhook with the generated secret"
    log_info "3. Configure external secrets operator for production secrets"
    log_info "4. Test automated deployment by pushing changes to the repository"
}

# Run main function
main "$@"
