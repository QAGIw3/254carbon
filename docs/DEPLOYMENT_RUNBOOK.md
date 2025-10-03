# Production Deployment Runbook

This runbook outlines the operational steps for Weeks 2 and 3 of the 254Carbon deployment roadmap:
service rollout and data pipeline activation.

## Week 2 – Service Deployment

### 1. Infrastructure Validation
- Confirm Week 1 provisioning completed (EKS clusters, Aurora Global DB, Redis global datastore).
- Validate DNS and certificates (Route53, CloudFront).
- Ensure `kubectl` contexts exist for each region (`254carbon-us-east`, `254carbon-eu-west`, `254carbon-apac`).

### 2. Deploy Shared Namespaces & Secrets
```bash
# For each cluster context
declare -a clusters=("254carbon-us-east" "254carbon-eu-west" "254carbon-apac")
for ctx in "${clusters[@]}"; do
  kubectl --context="$ctx" apply -f platform/infra/k8s/namespace.yaml
  kubectl --context="$ctx" apply -f platform/infra/k8s/network-policy.yaml
  kubectl --context="$ctx" apply -f platform/infra/k8s/security/network-policies.yaml
  kubectl --context="$ctx" apply -f platform/infra/k8s/security/ip-allowlist.yaml
  kubectl --context="$ctx" apply -f platform/infra/k8s/security/external-secrets.yaml
  kubectl --context="$ctx" apply -f platform/infra/k8s/secrets-rotation-cronjob.yaml
  kubectl --context="$ctx" apply -f platform/infra/k8s/pod-security-policy.yaml
  kubectl --context="$ctx" annotate namespace market-intelligence \
    "traffic.sidecar.istio.io/includeInboundPorts=*"
done
```

### 3. Deploy Infrastructure Services
```bash
# Helm chart per region
for ctx in "${clusters[@]}"; do
  kubectl --context="$ctx" create namespace data-layer || true
  helm upgrade --install clickhouse platform/infra/helm/clickhouse \
    --namespace data-layer --kube-context "$ctx"
  helm upgrade --install kafka platform/infra/helm/kafka \
    --namespace data-layer --kube-context "$ctx"
  helm upgrade --install minio platform/infra/helm/minio \
    --namespace data-layer --kube-context "$ctx"
  helm upgrade --install keycloak platform/infra/helm/keycloak \
    --namespace platform-auth --create-namespace --kube-context "$ctx"
  helm upgrade --install gateway platform/infra/helm/ingress \
    --namespace networking --create-namespace --kube-context "$ctx"
done

# Aurora migration scripts
psql "host=${AURORA_ENDPOINT} dbname=market_intelligence user=${DB_USER} password=${DB_PASS}" \
  -f platform/data/schemas/postgres/init.sql
```

### 4. Deploy Application Services
```bash
for ctx in "${clusters[@]}"; do
  helm upgrade --install gateway platform/apps/gateway \
    --namespace market-intelligence --kube-context "$ctx"
  helm upgrade --install curve-service platform/apps/curve-service \
    --namespace market-intelligence --kube-context "$ctx"
  helm upgrade --install scenario-engine platform/apps/scenario-engine \
    --namespace market-intelligence --kube-context "$ctx"
  helm upgrade --install backtesting-service platform/apps/backtesting-service \
    --namespace market-intelligence --kube-context "$ctx"
  helm upgrade --install download-center platform/apps/download-center \
    --namespace market-intelligence --kube-context "$ctx"
  helm upgrade --install report-service platform/apps/report-service \
    --namespace market-intelligence --kube-context "$ctx"
  helm upgrade --install web-hub platform/apps/web-hub \
    --namespace market-intelligence --kube-context "$ctx"
done
```

### 5. Monitoring & Observability
- Ensure Prometheus/Grafana helm releases installed (`platform/infra/monitoring/` charts).
- Import dashboard JSONs from `platform/infra/monitoring/grafana/dashboards/`.
- Configure Alertmanager routes and PagerDuty integration.

### 6. Validation Checklist
- [ ] `kubectl get pods -A` healthy in each cluster.
- [ ] `/health` endpoints return 200 for core services.
- [ ] Grafana dashboards display metrics across regions.
- [ ] Secrets rotation cronjob succeeds.

## Week 3 – Data Pipeline Activation

### 1. Kafka Topics & Schemas
```bash
# Example using kafka-topics.sh
kafka-topics.sh --bootstrap-server ${KAFKA_BROKER} --create \
  --topic market.price.ticks --replication-factor 3 --partitions 12

# Register Avro schema
platform/scripts/register-schema.sh platform/data/schemas/avro/market_price_ticks.avsc
```

### 2. Enable Airflow DAGs
```bash
# Update Airflow variables and connections
airflow variables set MISO_CONNECTOR_CONFIG @/tmp/miso_config.json
airflow connections add --conn-id clickhouse --conn-type clickhouse \
  --conn-host ${CLICKHOUSE_HOST} --conn-login ${CLICKHOUSE_USER} --conn-password ${CLICKHOUSE_PASS}

# Unpause DAGs
airflow dags unpause miso_ingestion_dag
airflow dags unpause caiso_ingestion_dag
```

### 3. Historical Backfill
```bash
python platform/scripts/seed-data.sh --market miso --days 30
python platform/scripts/seed-data.sh --market caiso --days 30
```

### 4. Data Quality Validation
- Check ClickHouse for completeness (>=99.5% expected rows).
- Verify data freshness alerts in Grafana.
- Inspect Airflow DAG run statuses and logs.

### 5. Backtesting Validation
```bash
python platform/apps/backtesting-service/main.py \
  --market miso --start-date 2025-08-01 --end-date 2025-09-30

curl -H "Authorization: Bearer ${TOKEN}" \
  "https://api.254carbon.ai/backtesting/results?market=MISO"
```
- Confirm MAPE ≤ 12%, WAPE ≤ 10%, RMSE within historical bounds.

### 6. Sign-off Checklist
- [ ] MISO real-time stream latency <2s across Web Hub.
- [ ] CAISO entitlement restrictions enforced (API blocked, downloads enabled).
- [ ] Backfill accuracy validated, metrics exported to Grafana.
- [ ] Support runbooks updated with contact/escalation info.

## Rollback Plan
1. Disable Airflow DAGs (`airflow dags pause ...`).
2. Scale deployments to zero (`kubectl scale deploy --replicas=0`).
3. Restore ClickHouse/PostgreSQL from latest snapshots.
4. Re-enable staging environment for pilot customers.

## Contacts
- **Infra On-Call**: PagerDuty rotation `infra-oncall`
- **Data Engineering**: `data-eng@254carbon.ai`
- **Security**: `security@254carbon.ai`
