# Infrastructure Deployment Guide

## Overview

This guide covers the deployment and management of the 254Carbon Market Intelligence Platform infrastructure using Terraform, Helm, and ArgoCD.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AWS Cloud     │    │   Kubernetes    │    │   Application   │
│                 │    │                 │    │                 │
│ • EKS Cluster   │◄──►│ • API Gateway   │◄──►│ • Curve Service │
│ • RDS PostgreSQL│    │ • Curve Service │    │ • Scenario Eng. │
│ • ElastiCache   │    │ • Backtesting   │    │ • Web Hub       │
│ • VPC/Networking│    │ • Monitoring    │    │ • Download Ctr. │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Deployment Scripts

### 1. Infrastructure Deployment (`deploy-production.sh`)

Deploy the complete infrastructure stack to AWS:

```bash
# Deploy to production
./deploy-production.sh prod

# Deploy to staging
./deploy-production.sh staging us-east-1

# Deploy to development (single AZ, reduced resources)
./deploy-production.sh dev us-east-1
```

**What it does:**
- Provisions EKS cluster with production-grade security
- Sets up RDS PostgreSQL with high availability
- Configures ElastiCache Redis cluster
- Deploys monitoring stack (Prometheus, Grafana)
- Applies network policies and security configurations

### 2. ArgoCD GitOps Setup (`deploy-argocd.sh`)

Set up continuous deployment with ArgoCD:

```bash
# Deploy ArgoCD for development
./deploy-argocd.sh argocd dev

# Deploy ArgoCD for production (requires approval)
./deploy-argocd.sh argocd prod
```

**What it does:**
- Installs ArgoCD in the cluster
- Creates application projects and RBAC
- Sets up automated deployment from Git
- Configures webhooks for CI/CD

### 3. Blue-Green Deployment (`switch-deployment.sh`)

Switch between blue and green deployments:

```bash
# Switch to green deployment
./switch-deployment.sh market-intelligence market-intelligence green

# Switch back to blue (automatic if no color specified)
./switch-deployment.sh market-intelligence market-intelligence
```

**What it does:**
- Scales down old deployment
- Scales up new deployment
- Updates service selectors
- Runs health checks
- Automatic rollback on failure

## Manual Deployment Steps

### 1. Database Initialization

After infrastructure deployment, initialize the databases:

```bash
# Initialize PostgreSQL schema
kubectl exec -n market-intelligence postgresql-0 -- \
  psql -U postgres -d market_intelligence -f /platform/data/schemas/postgres/init.sql

# Initialize ClickHouse schema
kubectl exec -n market-intelligence clickhouse-0 -- \
  clickhouse-client --query "source /platform/data/schemas/clickhouse/init.sql"
```

### 2. Secrets Management

Set up external secrets for production:

```bash
# Deploy external-secrets operator
kubectl apply -f platform/infra/k8s/security/external-secrets.yaml

# Configure AWS Secrets Manager backend
kubectl apply -f platform/infra/k8s/security/secret-store.yaml
```

### 3. Application Deployment

Deploy the application using Helm:

```bash
# Deploy to development
helm upgrade --install market-intelligence-dev platform/infra/helm/market-intelligence \
  --namespace market-intelligence-dev --create-namespace \
  --values platform/infra/helm/market-intelligence/values-dev.yaml

# Deploy to production with blue-green
helm upgrade --install market-intelligence platform/infra/helm/market-intelligence \
  --namespace market-intelligence --create-namespace \
  --values platform/infra/helm/market-intelligence/values-prod.yaml
```

## Monitoring & Observability

### Grafana Dashboards

Access Grafana at `https://your-domain/grafana` with admin credentials from Terraform output.

**Available Dashboards:**
- **Business KPIs**: Revenue, customer usage, market coverage
- **Forecast Accuracy**: MAPE, WAPE, RMSE by market and horizon
- **Data Quality**: Freshness, completeness, anomaly detection
- **Service Health**: Latency, error rates, resource usage
- **Security & Audit**: Authentication, authorization, data exports

### Prometheus Metrics

Key metrics to monitor:
- `254carbon_api_request_duration_seconds` - API latency
- `254carbon_forecast_mape` - Forecast accuracy
- `254carbon_data_completeness` - Data quality
- `254carbon_customer_usage` - Business metrics

### Alerting

Critical alerts configured:
- API latency > 250ms (p95)
- Data completeness < 99.5%
- Forecast accuracy degradation
- Service downtime

## Troubleshooting

### Common Issues

1. **EKS Node Group Issues**
   ```bash
   # Check node status
   kubectl get nodes -o wide

   # Check pod scheduling
   kubectl describe pod <pod-name> -n <namespace>
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connectivity
   kubectl exec -n market-intelligence postgresql-0 -- \
     psql -U postgres -d market_intelligence -c "SELECT 1;"

   # Check ClickHouse
   kubectl exec -n market-intelligence clickhouse-0 -- \
     clickhouse-client --query "SELECT 1;"
   ```

3. **Application Health**
   ```bash
   # Check application logs
   kubectl logs -n market-intelligence deployment/api-gateway

   # Check service endpoints
   kubectl get svc -n market-intelligence
   ```

### Rollback Procedures

1. **Application Rollback**
   ```bash
   # Using Helm
   helm rollback market-intelligence <revision-number>

   # Using blue-green switch
   ./switch-deployment.sh market-intelligence market-intelligence blue
   ```

2. **Infrastructure Rollback**
   ```bash
   cd platform/infra/terraform/environments/prod
   terraform plan  # Review changes
   terraform apply # Apply rollback
   ```

## Security Considerations

### Production Security

1. **Network Security**
   - Private subnets for application services
   - Security groups restrict traffic
   - Network policies enforce zero-trust

2. **Access Control**
   - IAM roles with minimal permissions
   - Kubernetes RBAC configured
   - Secrets encrypted at rest and in transit

3. **Compliance**
   - SOC2 controls implemented
   - Audit logging enabled
   - Data encryption standards met

### Certificate Management

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Issue Let's Encrypt certificate
kubectl apply -f platform/infra/k8s/security/cluster-issuer.yaml
```

## Cost Optimization

### AWS Cost Monitoring

1. **Resource Tagging**
   - All resources tagged with `Environment`, `Project`, `CostCenter`
   - Use Cost Explorer to track expenses

2. **Right-sizing**
   - Monitor resource utilization
   - Use HPA for auto-scaling
   - Consider spot instances for non-critical workloads

3. **Storage Optimization**
   - Use gp3 SSD for EBS volumes
   - Implement data retention policies
   - Archive old data to S3 Glacier

## Maintenance Procedures

### Regular Tasks

1. **Weekly**
   - Review Grafana dashboards for anomalies
   - Check data quality metrics
   - Review security audit logs

2. **Monthly**
   - Rotate secrets (automated via external-secrets)
   - Update SSL certificates
   - Review and optimize costs

3. **Quarterly**
   - Disaster recovery testing
   - Security vulnerability assessments
   - Performance benchmarking

### Backup Procedures

1. **Database Backups**
   - RDS automated snapshots (30-day retention)
   - ClickHouse manual snapshots before major changes

2. **Application Backups**
   - Helm release history maintained
   - Git repository serves as configuration backup

3. **Disaster Recovery**
   - Multi-AZ deployment for high availability
   - Automated failover testing quarterly

## Support & Escalation

### Issue Severity Levels

| Level | Response Time | Escalation |
|-------|---------------|------------|
| **P0 - Critical** | 15 minutes | On-call → Manager → CTO |
| **P1 - High** | 1 hour | On-call → Manager |
| **P2 - Medium** | 4 hours | On-call |
| **P3 - Low** | 24 hours | Support team |

### Contact Information

- **On-Call Engineering**: PagerDuty rotation
- **Security Issues**: security@254carbon.ai
- **Customer Support**: support@254carbon.ai
- **Executive Escalation**: CTO, CEO

## Next Steps

1. **Deploy to Staging**: Test complete pipeline before production
2. **Pilot Customer Onboarding**: Configure customer-specific settings
3. **Performance Optimization**: Tune database queries and caching
4. **Feature Development**: Begin Phase 2 enhancements

## Documentation References

- [Terraform Documentation](https://registry.terraform.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)

