#!/usr/bin/env bash
set -euo pipefail

: "${VAULT_ADDR:?set VAULT_ADDR}"
: "${VAULT_TOKEN:?set VAULT_TOKEN}"

vault login "$VAULT_TOKEN" >/dev/null

# Enable KV v2 at path secret/
vault secrets enable -path=secret -version=2 kv || true

# Enable Kubernetes auth
vault auth enable kubernetes || true

# Configure Kubernetes auth (run this from inside a pod in the cluster)
# Uses the service account token and CA of the current pod
if vault read auth/kubernetes/config >/dev/null 2>&1; then
  echo "Kubernetes auth already configured"
else
  vault write auth/kubernetes/config \
    token_reviewer_jwt=@/var/run/secrets/kubernetes.io/serviceaccount/token \
    kubernetes_host="https://${KUBERNETES_PORT_443_TCP_ADDR:-kubernetes.default}.svc:443" \
    kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
fi

# Write policies
vault policy write airflow policies/airflow.hcl
vault policy write openmetadata policies/openmetadata.hcl
vault policy write mlflow policies/mlflow.hcl
vault policy write nifi policies/nifi.hcl
vault policy write platform-secrets policies/platform-secrets.hcl

# Create roles bound to namespace data-platform (any SA)
for app in airflow openmetadata mlflow nifi platform-secrets; do
  vault write auth/kubernetes/role/${app} \
    bound_service_account_names="*" \
    bound_service_account_namespaces="data-platform" \
    policies="${app}" \
    ttl="1h"
done

echo "Bootstrap complete. Create secrets under secret/data/<app>/..."
