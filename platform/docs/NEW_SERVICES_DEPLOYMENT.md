# New Services Deployment Checklist

Comprehensive checklist for deploying newly implemented 254Carbon services to production.

## Pre-Deployment Checklist

### ✅ Development Complete

- [x] LMP Decomposition Service implemented
- [x] Trading Signals Service implemented
- [x] Marketplace Service implemented
- [x] Transformer ML models integrated
- [x] New market connectors (IESO, NEM, Brazil ONS)

### ✅ Testing & Quality

- [x] Integration tests written for all services
- [x] Load tests created for new endpoints
- [x] Unit test coverage >80%
- [x] API documentation complete
- [x] Code review completed

### ✅ Infrastructure

- [x] Dockerfiles created for all services
- [x] Kubernetes manifests created
- [x] Airflow DAGs created for new connectors
- [x] Health check endpoints implemented
- [x] Prometheus metrics added
- [x] Grafana dashboards created

### ✅ Security

- [x] Non-root containers configured
- [x] Security scanning scripts created
- [x] Network policies defined
- [x] Secrets management configured
- [x] Rate limiting configured

## Deployment Steps

### Week 1: Staging Deployment

#### Day 1-2: Build and Push Images

```bash
# Navigate to service directories and build Docker images
cd platform/apps/marketplace
docker build -t 254carbon/marketplace:v1.0.0 .
docker push 254carbon/marketplace:v1.0.0

cd platform/apps/signals-service
docker build -t 254carbon/signals-service:v1.0.0 .
docker push 254carbon/signals-service:v1.0.0

cd platform/apps/fundamental-models
docker build -t 254carbon/fundamental-models:v1.0.0 .
docker push 254carbon/fundamental-models:v1.0.0
```

#### Day 2-3: Deploy to Staging

```bash
# Deploy services to staging namespace
kubectl create namespace market-intelligence-staging

# Deploy marketplace
kubectl apply -f platform/apps/marketplace/k8s/deployment.yaml -n market-intelligence-staging

# Deploy signals service
kubectl apply -f platform/apps/signals-service/k8s/deployment.yaml -n market-intelligence-staging

# Deploy fundamental models
kubectl apply -f platform/apps/fundamental-models/k8s/deployment.yaml -n market-intelligence-staging

# Verify deployments
kubectl get pods -n market-intelligence-staging
kubectl get svc -n market-intelligence-staging
```

#### Day 3-4: Staging Validation

```bash
# Run smoke tests
cd platform/tests
./smoke-tests-staging.sh

# Run integration tests
pytest platform/apps/marketplace/tests/ -v
pytest platform/apps/signals-service/tests/ -v
pytest platform/apps/lmp-decomposition-service/tests/ -v

# Run load tests
cd platform/tests/load
k6 run new-services-load-test.js --env BASE_URL=https://staging.254carbon.ai
```

#### Day 4-5: Security Scanning

```bash
# Run security scans
cd platform/infra/scripts
chmod +x security-scan-new-services.sh
./security-scan-new-services.sh

# Review scan results
cat security-scan-results/*

# Fix any critical vulnerabilities
# Rebuild and redeploy if needed
```

### Week 2: Production Deployment

#### Day 1: Pre-Production Review

- [ ] All staging tests passed
- [ ] Security scans completed with no critical issues
- [ ] Load tests validated performance targets
- [ ] Grafana dashboards configured
- [ ] Alert rules configured in Prometheus
- [ ] Runbooks reviewed by operations team

#### Day 2: Production Deployment

```bash
# Deploy to production namespace
kubectl apply -f platform/apps/marketplace/k8s/deployment.yaml -n market-intelligence
kubectl apply -f platform/apps/signals-service/k8s/deployment.yaml -n market-intelligence
kubectl apply -f platform/apps/fundamental-models/k8s/deployment.yaml -n market-intelligence

# Verify deployments
kubectl get pods -n market-intelligence
kubectl rollout status deployment/marketplace -n market-intelligence
kubectl rollout status deployment/signals-service -n market-intelligence
kubectl rollout status deployment/fundamental-models -n market-intelligence
```

#### Day 2-3: Production Validation

```bash
# Run smoke tests
./smoke-tests-prod.sh

# Monitor dashboards
# - Open Grafana: http://254carbon.local/grafana
# - Check "New Services Performance" dashboard
# - Monitor error rates, latency, cache hit rates

# Verify metrics collection
curl http://marketplace:8015/metrics
curl http://signals-service:8016/metrics
```

#### Day 3-5: Deploy Airflow DAGs

```bash
# Copy new DAGs to Airflow
kubectl cp platform/data/ingestion-orch/dags/ieso_ingestion_dag.py airflow-scheduler:/opt/airflow/dags/
kubectl cp platform/data/ingestion-orch/dags/nem_ingestion_dag.py airflow-scheduler:/opt/airflow/dags/
kubectl cp platform/data/ingestion-orch/dags/brazil_ons_ingestion_dag.py airflow-scheduler:/opt/airflow/dags/

# Trigger DAGs manually for first run
airflow dags trigger ieso_hoep_ingestion
airflow dags trigger nem_spot_price_ingestion
airflow dags trigger brazil_pld_ingestion

# Monitor execution
airflow dags list
airflow tasks list ieso_hoep_ingestion
```

## Post-Deployment Verification

### Service Health Checks

```bash
# Check all new services are healthy
curl http://marketplace:8015/health
curl http://signals-service:8016/health
curl http://fundamental-models:8031/health
curl http://lmp-decomposition-service:8009/health
```

### Data Quality Checks

```bash
# Verify connectors are running
kubectl logs -n market-intelligence -l app=airflow-scheduler --tail=100

# Check ClickHouse for new data
clickhouse-client --query "SELECT COUNT(*) FROM ch.market_price_ticks WHERE source IN ('ieso', 'nem', 'ons')"

# Verify materialized views are populated
clickhouse-client --query "SELECT COUNT(*) FROM ch.hourly_price_aggregations WHERE market IN ('IESO', 'NEM', 'ONS')"
```

### Performance Validation

- [ ] API latency p95 < target (check Grafana)
- [ ] Cache hit rate > 70%
- [ ] Error rate < 1%
- [ ] Database query performance acceptable
- [ ] No resource constraint alerts

### Security Validation

- [ ] All security scans passed
- [ ] Network policies enforced
- [ ] Secrets properly configured
- [ ] Audit logging operational
- [ ] Rate limiting enforced

## Rollback Plan

If issues are detected:

```bash
# Rollback specific service
kubectl rollout undo deployment/marketplace -n market-intelligence

# Or rollback to specific revision
kubectl rollout history deployment/marketplace -n market-intelligence
kubectl rollout undo deployment/marketplace --to-revision=2 -n market-intelligence

# Disable problematic DAGs
airflow dags pause ieso_hoep_ingestion

# Verify rollback
kubectl get pods -n market-intelligence
kubectl rollout status deployment/marketplace -n market-intelligence
```

## Monitoring & Alerts

### Key Metrics to Monitor (First 48 Hours)

1. **Service Health**
   - All pods running and ready
   - No crash loops
   - Health checks passing

2. **Performance**
   - API latency within SLA
   - Cache hit rate > 70%
   - Database query performance

3. **Data Quality**
   - Connector ingestion successful
   - Data freshness within bounds
   - No data validation errors

4. **Security**
   - No failed authentication attempts spike
   - Rate limits not triggering excessively
   - No security scan alerts

### Alert Channels

- **PagerDuty**: Critical alerts (service down, data breach)
- **Slack #prod-alerts**: High priority alerts
- **Email ops@254carbon.ai**: Medium/low priority alerts
- **Grafana**: Visual monitoring and dashboards

## Success Criteria

### Technical Metrics (30 Days)

- **Uptime**: > 99.9%
- **API Latency p95**: < 250ms
- **Error Rate**: < 1%
- **Cache Hit Rate**: > 70%
- **Data Freshness**: Within configured bounds

### Business Metrics

- **Marketplace**: > 5 active partners
- **Trading Signals**: > 1000 signals generated
- **ML Forecasts**: > 500 forecasts generated
- **LMP Decomposition**: > 10000 decompositions calculated

## Support & Escalation

### On-Call Contacts

- **Primary**: DevOps Engineer (15min SLA)
- **Secondary**: Platform Lead (1hr SLA)
- **Escalation**: CTO (4hr SLA)

### Runbooks

- [LMP Decomposition Troubleshooting](./runbooks/lmp-decomposition.md)
- [Trading Signals Calibration](./runbooks/trading-signals.md)
- [Marketplace Operations](./runbooks/marketplace.md)
- [Transformer Model Retraining](./runbooks/transformer-retraining.md)

---

**Deployment Prepared By:** 254Carbon Engineering Team  
**Date:** October 4, 2025  
**Status:** Ready for Staging Deployment

