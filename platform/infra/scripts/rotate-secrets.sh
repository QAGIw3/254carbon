#!/bin/bash
# Automated Secrets Rotation Script
# This script rotates database passwords and API keys

set -euo pipefail

# Configuration
NAMESPACE="market-intelligence"
SECRETS_BACKEND="${SECRETS_BACKEND:-aws-secrets-manager}"  # aws-secrets-manager or vault
REGION="${AWS_REGION:-us-east-1}"

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

# Generate secure random password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Rotate PostgreSQL password
rotate_postgres_password() {
    log_info "Rotating PostgreSQL password..."
    
    local new_password=$(generate_password)
    local secret_name="prod/market-intelligence/postgres"
    
    # Update password in secrets manager
    if [ "$SECRETS_BACKEND" = "aws-secrets-manager" ]; then
        aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "{\"username\":\"postgres\",\"password\":\"$new_password\",\"connection_string\":\"postgresql://postgres:$new_password@postgres:5432/market_intelligence\"}" \
            --region "$REGION"
    elif [ "$SECRETS_BACKEND" = "vault" ]; then
        vault kv put secret/$secret_name \
            username=postgres \
            password="$new_password" \
            connection_string="postgresql://postgres:$new_password@postgres:5432/market_intelligence"
    fi
    
    # Update actual PostgreSQL password
    kubectl exec -n "$NAMESPACE" postgres-0 -- \
        psql -U postgres -c "ALTER USER postgres WITH PASSWORD '$new_password';"
    
    log_info "PostgreSQL password rotated successfully"
}

# Rotate ClickHouse password
rotate_clickhouse_password() {
    log_info "Rotating ClickHouse password..."
    
    local new_password=$(generate_password)
    local secret_name="prod/market-intelligence/clickhouse"
    
    # Update password in secrets manager
    if [ "$SECRETS_BACKEND" = "aws-secrets-manager" ]; then
        aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "{\"username\":\"default\",\"password\":\"$new_password\"}" \
            --region "$REGION"
    elif [ "$SECRETS_BACKEND" = "vault" ]; then
        vault kv put secret/$secret_name \
            username=default \
            password="$new_password"
    fi
    
    # Update ClickHouse password
    kubectl exec -n "$NAMESPACE" clickhouse-0 -- \
        clickhouse-client --query "ALTER USER default IDENTIFIED BY '$new_password'"
    
    log_info "ClickHouse password rotated successfully"
}

# Rotate Keycloak admin password
rotate_keycloak_password() {
    log_info "Rotating Keycloak admin password..."
    
    local new_password=$(generate_password)
    local secret_name="prod/market-intelligence/keycloak"
    
    # Update password in secrets manager
    if [ "$SECRETS_BACKEND" = "aws-secrets-manager" ]; then
        aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "{\"admin_username\":\"admin\",\"admin_password\":\"$new_password\",\"client_secret\":\"$(generate_password)\"}" \
            --region "$REGION"
    elif [ "$SECRETS_BACKEND" = "vault" ]; then
        vault kv put secret/$secret_name \
            admin_username=admin \
            admin_password="$new_password" \
            client_secret="$(generate_password)"
    fi
    
    log_info "Keycloak admin password rotated in secrets manager"
    log_warn "Manual update required in Keycloak admin console"
}

# Rotate MinIO access keys
rotate_minio_keys() {
    log_info "Rotating MinIO access keys..."
    
    local new_access_key="AKIA$(openssl rand -hex 8 | tr '[:lower:]' '[:upper:]')"
    local new_secret_key=$(generate_password)
    local secret_name="prod/market-intelligence/minio"
    
    # Update keys in secrets manager
    if [ "$SECRETS_BACKEND" = "aws-secrets-manager" ]; then
        aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "{\"access_key\":\"$new_access_key\",\"secret_key\":\"$new_secret_key\"}" \
            --region "$REGION"
    elif [ "$SECRETS_BACKEND" = "vault" ]; then
        vault kv put secret/$secret_name \
            access_key="$new_access_key" \
            secret_key="$new_secret_key"
    fi
    
    log_info "MinIO keys rotated successfully"
}

# Restart services to pick up new secrets
restart_services() {
    log_info "Restarting services to pick up new secrets..."
    
    # Restart API Gateway
    kubectl rollout restart deployment/api-gateway -n "$NAMESPACE"
    kubectl rollout status deployment/api-gateway -n "$NAMESPACE" --timeout=300s
    
    # Restart Curve Service
    kubectl rollout restart deployment/curve-service -n "$NAMESPACE"
    kubectl rollout status deployment/curve-service -n "$NAMESPACE" --timeout=300s
    
    # Restart Backtesting Service
    kubectl rollout restart deployment/backtesting-service -n "$NAMESPACE"
    kubectl rollout status deployment/backtesting-service -n "$NAMESPACE" --timeout=300s
    
    log_info "All services restarted successfully"
}

# Verify rotation success
verify_rotation() {
    log_info "Verifying secret rotation..."
    
    # Check that External Secrets are synced
    local sync_status=$(kubectl get externalsecrets -n "$NAMESPACE" -o json | \
        jq -r '.items[] | select(.status.conditions[].status == "False") | .metadata.name')
    
    if [ -n "$sync_status" ]; then
        log_error "Some External Secrets failed to sync: $sync_status"
        return 1
    fi
    
    # Check service health
    local unhealthy=$(kubectl get pods -n "$NAMESPACE" -o json | \
        jq -r '.items[] | select(.status.phase != "Running") | .metadata.name')
    
    if [ -n "$unhealthy" ]; then
        log_error "Some pods are not healthy: $unhealthy"
        return 1
    fi
    
    log_info "Verification passed - all secrets rotated successfully"
}

# Main rotation workflow
main() {
    log_info "Starting automated secrets rotation..."
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    if [ "$SECRETS_BACKEND" = "aws-secrets-manager" ] && ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
    
    if [ "$SECRETS_BACKEND" = "vault" ] && ! command -v vault &> /dev/null; then
        log_error "Vault CLI is not installed"
        exit 1
    fi
    
    # Perform rotation
    rotate_postgres_password
    sleep 5
    
    rotate_clickhouse_password
    sleep 5
    
    rotate_keycloak_password
    sleep 5
    
    rotate_minio_keys
    sleep 5
    
    # Wait for External Secrets to sync
    log_info "Waiting for External Secrets to sync (60s)..."
    sleep 60
    
    # Restart services
    restart_services
    
    # Verify
    verify_rotation
    
    log_info "Secrets rotation completed successfully!"
}

# Run main function
main "$@"



