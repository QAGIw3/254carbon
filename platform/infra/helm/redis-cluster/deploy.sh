#!/bin/bash
# Deploy Redis Cluster for 254Carbon

set -e

echo "=== Deploying Redis Cluster ==="

# Create namespace
kubectl create namespace market-intelligence-infra --dry-run=client -o yaml | kubectl apply -f -

# Generate Redis password
REDIS_PASSWORD=$(openssl rand -base64 32)

# Create secret
kubectl create secret generic redis-password \
  --namespace market-intelligence-infra \
  --from-literal=password=$REDIS_PASSWORD \
  --dry-run=client -o yaml | kubectl apply -f -

# Add Redis Helm repo
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Deploy Redis cluster
helm upgrade --install redis-cluster bitnami/redis \
  --namespace market-intelligence-infra \
  --values values.yaml \
  --set auth.password=$REDIS_PASSWORD

echo ""
echo "=== Redis Cluster Deployed ==="
echo ""
echo "Connection details:"
echo "  Host: redis-cluster-master.market-intelligence-infra.svc.cluster.local"
echo "  Port: 6379"
echo "  Password: (stored in secret redis-password)"
echo ""
echo "To get password:"
echo "  kubectl get secret redis-password -n market-intelligence-infra -o jsonpath='{.data.password}' | base64 -d"
echo ""
echo "To test connection:"
echo "  kubectl run redis-test --rm -it --image redis:7 -- redis-cli -h redis-cluster-master.market-intelligence-infra.svc.cluster.local -a \$REDIS_PASSWORD"

