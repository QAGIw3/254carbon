#!/bin/bash

# 254Carbon Production Infrastructure Deployment Script
# This script deploys the complete infrastructure stack to production

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-prod}
REGION=${2:-us-east-1}
TERRAFORM_DIR="../terraform/environments/${ENVIRONMENT}"

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

# Prerequisites check
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if AWS CLI is configured
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS CLI not configured. Please run 'aws configure' first."
        exit 1
    fi

    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform not installed. Please install Terraform first."
        exit 1
    fi

    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not installed. Please install kubectl first."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Initialize Terraform
init_terraform() {
    log_info "Initializing Terraform in ${TERRAFORM_DIR}..."
    cd "${TERRAFORM_DIR}"

    if [ ! -d ".terraform" ]; then
        terraform init
    else
        log_warn "Terraform already initialized, skipping init"
    fi

    log_success "Terraform initialized"
}

# Plan infrastructure
plan_infrastructure() {
    log_info "Planning infrastructure deployment..."
    cd "${TERRAFORM_DIR}"

    terraform plan \
        -var "aws_region=${REGION}" \
        -var "environment=${ENVIRONMENT}" \
        -out=tfplan

    log_success "Infrastructure plan created"
}

# Apply infrastructure
apply_infrastructure() {
    log_info "Applying infrastructure changes..."
    cd "${TERRAFORM_DIR}"

    terraform apply tfplan

    log_success "Infrastructure deployed successfully"
}

# Configure kubectl
configure_kubectl() {
    log_info "Configuring kubectl for EKS cluster..."
    cd "${TERRAFORM_DIR}"

    # Update kubeconfig with EKS cluster details
    terraform output -raw kubeconfig > ~/.kube/254carbon-${ENVIRONMENT}.yaml

    # Set the context
    kubectl config use-context $(kubectl config get-contexts -o name | grep 254carbon-${ENVIRONMENT})

    log_success "kubectl configured for EKS cluster"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    cd "${TERRAFORM_DIR}"

    # Deploy Prometheus and Grafana
    kubectl apply -f ../../monitoring/

    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-prometheus -n monitoring
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n monitoring

    log_success "Monitoring stack deployed"
}

# Deploy application services
deploy_application() {
    log_info "Deploying 254Carbon application services..."
    cd "${TERRAFORM_DIR}"

    # Deploy Helm charts
    helm upgrade --install 254carbon ../../helm/market-intelligence \
        --namespace market-intelligence --create-namespace \
        --values ../../helm/market-intelligence/values-${ENVIRONMENT}.yaml

    # Wait for key services to be ready
    kubectl wait --for=condition=available --timeout=600s deployment/api-gateway -n market-intelligence
    kubectl wait --for=condition=available --timeout=600s deployment/curve-service -n market-intelligence

    log_success "Application services deployed"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."

    # Check EKS cluster health
    kubectl get nodes
    kubectl get pods -A | grep -E "(Running|Completed)" | wc -l

    # Check application health
    kubectl exec -n market-intelligence deployment/api-gateway -- curl -f http://localhost:8000/health || {
        log_error "API Gateway health check failed"
        exit 1
    }

    log_success "All health checks passed"
}

# Main deployment flow
main() {
    log_info "Starting 254Carbon production deployment for ${ENVIRONMENT} environment"

    check_prerequisites
    init_terraform
    plan_infrastructure

    # Ask for confirmation before applying
    read -p "Do you want to proceed with the deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled"
        exit 0
    fi

    apply_infrastructure
    configure_kubectl
    deploy_monitoring
    deploy_application
    run_health_checks

    log_success "🎉 254Carbon production deployment completed successfully!"
    log_info "Next steps:"
    log_info "1. Configure DNS records to point to the load balancer"
    log_info "2. Set up SSL certificates"
    log_info "3. Configure external secrets backend"
    log_info "4. Activate data connectors"
    log_info "5. Run UAT with pilot customers"
}

# Run main function
main "$@"
