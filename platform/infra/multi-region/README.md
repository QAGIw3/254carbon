# Multi-Region Active-Active Deployment

Terraform configuration (`platform/infra/multi-region/terraform`) provisions the global production
footprint for the 254Carbon platform. The design delivers high availability, low-latency access, and
disaster recovery across three regions (US-East primary, EU-West, APAC).

## Architecture Summary

| Layer | Global Components |
|-------|-------------------|
| **Compute** | Amazon EKS clusters in us-east-1, eu-west-1, ap-southeast-1 |
| **Data** | Aurora PostgreSQL Global Database, ElastiCache Redis Global Datastore, regional ClickHouse clusters |
| **Routing/CDN** | Route53 latency-based routing + health checks, AWS CloudFront with multi-origin failover |
| **Security** | Regional KMS keys, IP allowlisting, network policies, WAF rules, automated secrets rotation |

### Traffic Flow
```
Users → CloudFront Edge → Route53 Latency Routing → Regional EKS → Microservices → Data Stores
```

## Terraform Layout

```
platform/infra/multi-region/terraform/
├── main.tf                # Root configuration (providers, modules, shared resources)
├── variables.tf           # Input variables for root module
└── modules/
    ├── eks/               # EKS cluster module (VPC, node groups, IAM)
    └── aurora-global/     # Aurora global database module
        └── secondary/     # Helper module for secondary regions
```

### Modules
- `modules/eks`: Provisions VPC, subnets, NAT gateways, IAM roles, EKS control plane, and managed node groups (supports GPU pools).
- `modules/aurora-global`: Creates Aurora global cluster with primary writer and secondary read replicas, networking, and security groups.
- `modules/aurora-global/secondary`: Regional VPC, subnet groups, and reader clusters for each replica region.

## Deployment Workflow

1. **Configure Terraform variables**
   - `db_master_username`, `db_master_password`
   - `db_kms_key_arn`
   - `acm_certificate_arn`
   - `provider_role_arn`
   - `regional_api_endpoints` (map of region → `{ domain_name, hosted_zone_id }`)
2. **Initialize and apply**
```bash
cd platform/infra/multi-region/terraform
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```
3. **Deploy services per region** using Helm once clusters are ready.

## Operational Guidance
- **Monitoring**: Use existing Grafana dashboards (service health, data quality, security) and CloudWatch metrics per region.
- **Failover**: CloudFront origin group fails over from US-East to EU-West; Route53 latency routing directs traffic to healthy origins.
- **Data replication**: Aurora global DB replicates <1s lag; Redis Global Datastore provides multi-active cache.
- **Maintenance windows** are staggered to keep at least two regions fully available.

## Next Steps & Enhancements
- Implement Terraform for ElastiCache regional replication groups and health checks feeding Route53.
- Add CI/CD automation (GitHub Actions/GitLab CI) to validate `terraform plan` on pull requests.
- Expand observability with synthetic latency checks per region.

**Last Updated:** 2025-10-03
