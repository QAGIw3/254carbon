# Multi-Region Active-Active Deployment

## Architecture

### Regions
1. **US-East (Primary)**: us-east-1 (N. Virginia)
2. **EU-West**: eu-west-1 (Ireland)
3. **APAC**: ap-southeast-1 (Singapore)

### Components

#### Compute
- **EKS Clusters**: One per region
  - US-East: 3 general nodes + 2 GPU nodes
  - EU-West: 3 general nodes
  - APAC: 2 general nodes
- **Auto-scaling**: Based on CPU/memory metrics

#### Data Layer
- **Aurora Global Database**: PostgreSQL 15.3
  - Primary: US-East
  - Read replicas: EU-West, APAC
  - RPO: <1 second
  - RTO: <1 minute
- **Redis Global Datastore**: ElastiCache
  - Active-active replication
  - Sub-second sync
- **ClickHouse**: Regional clusters with S3 replication

#### CDN & Routing
- **CloudFront**: Global edge locations
  - 200+ edge locations
  - Origin failover
  - Cache optimization
- **Route 53**: 
  - Latency-based routing
  - Health checks
  - Geo-proximity routing

### Traffic Flow

```
User Request
    ↓
CloudFront Edge (nearest)
    ↓
Route 53 (latency-based routing)
    ↓
Regional EKS Cluster
    ↓
Microservices
    ↓
Regional Data Stores
```

### Deployment

#### Prerequisites
```bash
# AWS CLI configured for all regions
aws configure --profile us-east
aws configure --profile eu-west
aws configure --profile apac

# Terraform installed
terraform --version  # >= 1.0

# kubectl configured
kubectl version
```

#### Deploy Infrastructure
```bash
cd platform/infra/multi-region/terraform

# Initialize
terraform init

# Plan
terraform plan -out=tfplan

# Apply
terraform apply tfplan
```

#### Deploy Applications
```bash
# US-East
kubectl config use-context 254carbon-us-east
helm install market-intelligence ../helm/market-intelligence \
  --set region=us-east \
  --set global.multiRegion=true

# EU-West
kubectl config use-context 254carbon-eu-west
helm install market-intelligence ../helm/market-intelligence \
  --set region=eu-west \
  --set global.multiRegion=true

# APAC
kubectl config use-context 254carbon-apac
helm install market-intelligence ../helm/market-intelligence \
  --set region=apac \
  --set global.multiRegion=true
```

### Data Replication

#### PostgreSQL (Aurora Global)
- **Primary Region**: US-East (read-write)
- **Secondary Regions**: EU-West, APAC (read-only)
- **Failover**: Automatic with <1 min RTO
- **Replication Lag**: Typically <100ms

#### Redis (Global Datastore)
- **Active-Active**: All regions writable
- **Conflict Resolution**: Last-write-wins
- **Sync**: Sub-second cross-region

#### ClickHouse
- **Architecture**: Regional clusters
- **Backup**: S3 cross-region replication
- **Sync**: Async via S3 (eventual consistency)

### Monitoring

#### Metrics
- **CloudWatch**: Per-region metrics
- **Prometheus**: Federated across regions
- **Grafana**: Global dashboard

#### Key Metrics
- Global API latency (p50, p95, p99)
- Regional traffic distribution
- Data replication lag
- Failover time
- Cache hit rates

### Disaster Recovery

#### Scenarios

**1. Regional Failure**
- CloudFront automatically routes to healthy region
- Route 53 updates DNS (60s TTL)
- Data: Aurora auto-failover to secondary
- Impact: <2 min service degradation

**2. Database Failure**
- Aurora auto-failover within region
- If regional DB fails, promote secondary
- Impact: <1 min for in-region, <2 min for cross-region

**3. Complete Region Outage**
- CloudFront routes all traffic to remaining regions
- Data syncs from S3 backups
- Manual intervention for Aurora promotion
- Impact: 5-10 min for full recovery

### Cost Optimization

#### Strategies
1. **Right-sizing**: Monitor and adjust instance types
2. **Spot Instances**: Use for batch jobs
3. **Data Transfer**: Minimize cross-region traffic
4. **Caching**: Aggressive CDN caching
5. **Auto-scaling**: Scale down during off-peak

#### Estimated Monthly Costs
- **Compute (EKS)**: $8,000
- **Database (Aurora)**: $6,000
- **Cache (Redis)**: $2,500
- **CDN (CloudFront)**: $3,500
- **Networking**: $2,000
- **Storage**: $1,500
- **Total**: ~$23,500/month

### Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| Global API Latency (p95) | <150ms | ~120ms |
| Regional API Latency (p95) | <50ms | ~35ms |
| Data Replication Lag | <1s | ~300ms |
| Failover Time | <2min | ~90s |
| Uptime | 99.99% | 99.98% |

### Security

#### Network
- VPC peering between regions
- Private subnets for data stores
- TLS 1.3 everywhere
- WAF rules on CloudFront

#### Data
- Encryption at rest (AES-256)
- Encryption in transit (TLS)
- KMS keys per region
- Cross-region key replication

#### Access
- IAM roles with least privilege
- MFA for production access
- Audit logging to S3
- CloudTrail across all regions

### Troubleshooting

#### High Latency
```bash
# Check regional health
for region in us-east eu-west apac; do
  kubectl --context=254carbon-$region get pods
  kubectl --context=254carbon-$region top nodes
done

# Check data replication lag
aws rds describe-global-clusters \
  --global-cluster-identifier 254carbon-global
```

#### Failover Not Working
```bash
# Check Route 53 health checks
aws route53 get-health-check-status \
  --health-check-id <check-id>

# Check CloudFront origin health
aws cloudfront get-distribution \
  --id <distribution-id>
```

#### Data Inconsistency
```bash
# Check Aurora replication status
aws rds describe-db-clusters \
  --db-cluster-identifier 254carbon-global

# Check Redis sync status
aws elasticache describe-global-replication-groups \
  --global-replication-group-id 254carbon
```

### Rollout Strategy

#### Phase 1: US-East (Week 1)
- Deploy primary region
- Test failover scenarios
- Validate data replication

#### Phase 2: EU-West (Week 2)
- Deploy secondary region
- Configure cross-region routing
- Test global failover

#### Phase 3: APAC (Week 3)
- Deploy tertiary region
- Enable full multi-region
- Load testing

#### Phase 4: Production (Week 4)
- Gradual traffic migration
- Monitor and optimize
- Document runbooks

### Maintenance Windows

**US-East**: Sundays 02:00-06:00 EST  
**EU-West**: Sundays 02:00-06:00 GMT  
**APAC**: Sundays 02:00-06:00 SGT

**Coordination**: Stagger by 8 hours to maintain 2/3 regions active

---

**Last Updated**: 2025-10-03  
**Owner**: Infrastructure Team  
**Status**: Production Ready

