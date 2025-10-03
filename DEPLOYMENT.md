# Deployment Guide

## Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (local: kind, k3d, or minikube)
- kubectl CLI
- Helm 3.x
- Python 3.11+
- Node.js 18+
- GNU Make

## Quick Start (Docker Compose)

For local development, use Docker Compose:

```bash
# Start all services
cd platform
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f gateway

# Access services
# - API Gateway: http://localhost:8000
# - Web Hub: http://localhost:3000
# - Keycloak: http://localhost:8080
# - MinIO Console: http://localhost:9001
```

## Production Deployment (Kubernetes)

### 1. Create Kubernetes Cluster

```bash
# Using kind
kind create cluster --name 254carbon

# Or using k3d
k3d cluster create 254carbon
```

### 2. Deploy Infrastructure

```bash
# Deploy infrastructure services
make infra

# Wait for services to be ready
kubectl get pods -n market-intelligence-infra -w
```

### 3. Initialize Databases

```bash
# Initialize schemas
make init-db
```

### 4. Deploy Application Services

```bash
# Build Docker images
docker build -t 254carbon/gateway:latest platform/apps/gateway
docker build -t 254carbon/curve-service:latest platform/apps/curve-service
docker build -t 254carbon/scenario-engine:latest platform/apps/scenario-engine
docker build -t 254carbon/download-center:latest platform/apps/download-center
docker build -t 254carbon/report-service:latest platform/apps/report-service

# Deploy via Helm
helm upgrade --install market-intelligence platform/infra/helm/market-intelligence \
  --namespace market-intelligence \
  --create-namespace
```

### 5. Configure Keycloak

```bash
# Port forward Keycloak
kubectl port-forward -n market-intelligence-infra svc/keycloak 8080:8080

# Access Keycloak admin console
# URL: http://localhost:8080
# User: admin
# Password: admin

# Create realm: 254carbon
# Create client: web-hub
# Configure OIDC settings
```

### 6. Access Platform

```bash
# Port forward gateway
kubectl port-forward -n market-intelligence svc/api-gateway 8000:8000

# Port forward web hub
kubectl port-forward -n market-intelligence svc/web-hub 3000:3000

# Access at http://localhost:3000
```

## Monitoring

### Prometheus

```bash
kubectl port-forward -n market-intelligence-infra svc/prometheus-server 9090:9090
# Access: http://localhost:9090
```

### Grafana

```bash
kubectl port-forward -n market-intelligence-infra svc/grafana 3001:80
# Access: http://localhost:3001
# User: admin
# Password: admin
```

## Data Ingestion

### Configure Source Connectors

```bash
# Register MISO connector
kubectl exec -n market-intelligence postgresql-0 -- psql -U postgres -d market_intelligence -c "
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('miso_rt_lmp', 'iso', 'https://api.misoenergy.org', 'active', 
        '{\"market_type\": \"RT\", \"kafka_topic\": \"power.ticks.v1\"}');
"

# Start connector
kubectl create job -n market-intelligence miso-connector \
  --image=254carbon/connectors:latest \
  -- python -m connectors.miso_connector
```

### Monitor Ingestion

```bash
# Check Kafka topics
kubectl exec -n market-intelligence-infra kafka-0 -- kafka-topics.sh --list --bootstrap-server localhost:9092

# Check ClickHouse data
kubectl exec -n market-intelligence-infra clickhouse-0 -- clickhouse-client --query "
SELECT COUNT(*) FROM ch.market_price_ticks;
"
```

## Entitlements Configuration

### MISO Pilot (Full Access)

```sql
INSERT INTO pg.entitlement_product (tenant_id, market, product, channels, seats)
VALUES ('pilot_miso', 'power', 'lmp', 
        '{"hub": true, "api": true, "downloads": true}'::jsonb, 5);
```

### CAISO Pilot (Hub + Downloads Only)

```sql
INSERT INTO pg.entitlement_product (tenant_id, market, product, channels, seats)
VALUES ('pilot_caiso', 'power', 'lmp', 
        '{"hub": true, "api": false, "downloads": true}'::jsonb, 3);
```

## Backup and Recovery

### PostgreSQL Backup

```bash
kubectl exec -n market-intelligence-infra postgresql-0 -- \
  pg_dump -U postgres market_intelligence > backup.sql
```

### ClickHouse Backup

```bash
kubectl exec -n market-intelligence-infra clickhouse-0 -- \
  clickhouse-client --query "BACKUP DATABASE ch TO Disk('backups', 'backup.zip')"
```

### MinIO Backup

```bash
mc mirror minio/raw s3://backup-bucket/raw
mc mirror minio/curves s3://backup-bucket/curves
```

## Troubleshooting

### Check Service Health

```bash
# API Gateway
curl http://localhost:8000/health

# Curve Service
curl http://localhost:8001/health

# Scenario Engine
curl http://localhost:8002/health
```

### View Logs

```bash
# Gateway logs
kubectl logs -n market-intelligence -l app=api-gateway -f

# Database logs
kubectl logs -n market-intelligence-infra postgresql-0 -f
kubectl logs -n market-intelligence-infra clickhouse-0 -f
```

### Performance Metrics

```bash
# API latency
curl http://localhost:9090/api/v1/query?query=api_request_duration_seconds

# Stream latency
curl http://localhost:9090/api/v1/query?query=stream_latency_seconds
```

## Scaling

### Horizontal Pod Autoscaling

```bash
kubectl autoscale deployment gateway -n market-intelligence \
  --cpu-percent=70 --min=2 --max=10
```

### ClickHouse Sharding

Configure sharding in `values.yaml`:

```yaml
clickhouse:
  shards: 3
  replicaCount: 2
```

## Security

### TLS Certificates

```bash
# Generate self-signed cert (dev only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt

# Create secret
kubectl create secret tls 254carbon-tls -n market-intelligence \
  --cert=tls.crt --key=tls.key
```

### Network Policies

Network policies are automatically applied. Verify:

```bash
kubectl get networkpolicies -n market-intelligence
```

### Secrets Rotation

```bash
# Rotate database passwords
kubectl create secret generic db-credentials -n market-intelligence \
  --from-literal=password=$(openssl rand -base64 32) \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new secrets
kubectl rollout restart deployment -n market-intelligence
```

## CI/CD Pipeline

The GitLab CI pipeline automatically:

1. Lints code (Python + TypeScript)
2. Runs unit tests
3. Builds Docker images
4. Scans for security vulnerabilities
5. Deploys to staging
6. Manual promotion to production

Configure GitLab variables:

- `KUBECONFIG`: Base64-encoded kubeconfig
- `DOCKER_REGISTRY`: Docker registry URL
- `DOCKER_USERNAME`: Registry username
- `DOCKER_PASSWORD`: Registry password

