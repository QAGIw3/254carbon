# Aurora Global Database Module

Terraform module that provisions an Amazon Aurora Global Database with a primary writer cluster and
secondary read replicas across multiple regions. Provides VPC networking, security groups, subnet
configuration, and cluster instances for each region.

## Inputs

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `global_cluster_identifier` | Global cluster identifier | string | n/a |
| `engine` | Database engine | string | `aurora-postgresql` |
| `engine_version` | Engine version | string | `15.3` |
| `database_name` | Database name | string | n/a |
| `primary_region` | Primary AWS region | string | n/a |
| `secondary_regions` | List of secondary regions | list(string) | `[]` |
| `master_username` | Master username | string | n/a |
| `master_password` | Master password | string | n/a |
| `kms_key_id` | KMS key ARN | string | n/a |
| `instance_class` | DB instance class | string | `db.r6g.xlarge` |
| `primary_instance_count` | Writer instances count | number | `2` |
| `secondary_instance_count` | Reader instances per secondary region | number | `1` |
| `primary_vpc_cidr` | VPC CIDR for primary region | string | n/a |
| `primary_availability_zones` | AZs for primary region | list(string) | n/a |
| `primary_private_subnets` | Private subnets for primary VPC | list(string) | n/a |
| `primary_public_subnets` | Public subnets for primary VPC | list(string) | n/a |
| `primary_allowed_cidrs` | CIDRs allowed to connect | list(string) | `[]` |
| `secondary_vpc_cidrs` | Map region -> VPC CIDR | map(string) | `{}` |
| `secondary_availability_zones` | Map region -> AZs | map(list(string)) | `{}` |
| `secondary_private_subnets` | Map region -> private subnets | map(list(string)) | `{}` |
| `secondary_public_subnets` | Map region -> public subnets | map(list(string)) | `{}` |
| `secondary_allowed_cidrs` | Map region -> allowed CIDRs | map(list(string)) | `{}` |
| `backup_retention_days` | Backup retention in days | number | `7` |
| `preferred_backup_window` | Backup window | string | `03:00-05:00` |
| `preferred_maintenance_window` | Maintenance window | string | `sun:05:00-sun:09:00` |
| `parameter_group_family` | Parameter group family | string | `aurora-postgresql15` |
| `db_timezone` | Database timezone | string | `UTC` |
| `tags` | Resource tags | map(string) | `{}` |
| `deletion_protection` | Enable deletion protection | bool | `true` |

## Outputs

- `primary_endpoint`
- `reader_endpoint`
- `arn`
- `cluster_id`

## Example

```hcl
module "aurora_global" {
  source = "./modules/aurora-global"

  global_cluster_identifier = "254carbon-global"
  database_name             = "market_intelligence"
  primary_region            = "us-east-1"
  secondary_regions         = ["eu-west-1", "ap-southeast-1"]
  master_username           = var.db_user
  master_password           = var.db_password
  kms_key_id                = aws_kms_key.db.arn

  primary_vpc_cidr            = "10.0.0.0/16"
  primary_availability_zones  = ["us-east-1a", "us-east-1b", "us-east-1c"]
  primary_private_subnets     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  primary_public_subnets      = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  primary_allowed_cidrs       = ["10.0.0.0/8"]

  tags = {
    Environment = "production"
    Project     = "254carbon"
  }
}
```
