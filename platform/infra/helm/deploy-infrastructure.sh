#!/bin/bash
# Deploy infrastructure services to local Kubernetes cluster

set -e

echo "=== Deploying 254Carbon Infrastructure ==="

# Create namespaces
kubectl apply -f ../k8s/namespace.yaml

# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add codecentric https://codecentric.github.io/helm-charts
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add apache-airflow https://airflow.apache.org
helm repo update

# Deploy PostgreSQL
echo "Deploying PostgreSQL..."
helm upgrade --install postgresql bitnami/postgresql \
  --namespace market-intelligence-infra \
  --set auth.postgresPassword=postgres \
  --set auth.database=market_intelligence \
  --set primary.persistence.size=20Gi

# Deploy ClickHouse
echo "Deploying ClickHouse..."
helm upgrade --install clickhouse bitnami/clickhouse \
  --namespace market-intelligence-infra \
  --set shards=1 \
  --set replicaCount=1 \
  --set persistence.size=50Gi

# Deploy Kafka
echo "Deploying Kafka..."
helm upgrade --install kafka bitnami/kafka \
  --namespace market-intelligence-infra \
  --set replicaCount=1 \
  --set persistence.size=10Gi

# Deploy MinIO
echo "Deploying MinIO..."
helm upgrade --install minio bitnami/minio \
  --namespace market-intelligence-infra \
  --set auth.rootUser=minioadmin \
  --set auth.rootPassword=minioadmin \
  --set persistence.size=100Gi

# Deploy Keycloak
echo "Deploying Keycloak..."
helm upgrade --install keycloak codecentric/keycloak \
  --namespace market-intelligence-infra \
  --set auth.adminUser=admin \
  --set auth.adminPassword=admin

# Deploy Prometheus
echo "Deploying Prometheus..."
helm upgrade --install prometheus prometheus-community/prometheus \
  --namespace market-intelligence-infra \
  --set server.retention=15d \
  --set server.persistentVolume.size=20Gi

# Deploy Grafana
echo "Deploying Grafana..."
helm upgrade --install grafana grafana/grafana \
  --namespace market-intelligence-infra \
  --set adminPassword=admin \
  --set persistence.enabled=true \
  --set persistence.size=5Gi

# Deploy Airflow
echo "Deploying Apache Airflow..."
helm upgrade --install airflow apache-airflow/airflow \
  --namespace market-intelligence-infra \
  --set webserver.defaultUser.username=admin \
  --set webserver.defaultUser.password=admin

# Apply network policies
kubectl apply -f ../k8s/network-policy.yaml

echo "=== Infrastructure deployment complete ==="
echo ""
echo "Access services:"
echo "  PostgreSQL:  kubectl port-forward -n market-intelligence-infra svc/postgresql 5432:5432"
echo "  ClickHouse:  kubectl port-forward -n market-intelligence-infra svc/clickhouse 8123:8123"
echo "  Kafka:       kubectl port-forward -n market-intelligence-infra svc/kafka 9092:9092"
echo "  MinIO:       kubectl port-forward -n market-intelligence-infra svc/minio 9000:9000"
echo "  Keycloak:    kubectl port-forward -n market-intelligence-infra svc/keycloak 8080:8080"
echo "  Prometheus:  kubectl port-forward -n market-intelligence-infra svc/prometheus-server 9090:9090"
echo "  Grafana:     kubectl port-forward -n market-intelligence-infra svc/grafana 3001:80"
echo "  Airflow:     kubectl port-forward -n market-intelligence-infra svc/airflow-webserver 8081:8080"

