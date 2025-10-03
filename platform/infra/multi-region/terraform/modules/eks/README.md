# Amazon EKS Module

Terraform module that provisions a production-grade Amazon EKS cluster with supporting VPC networking,
managed node groups (including optional GPU support), IAM roles, and security configuration. The module
is used by the multi-region deployment to create regional Kubernetes clusters.

## Inputs

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `region` | AWS region for the cluster | string | n/a |
| `cluster_name` | EKS cluster name | string | n/a |
| `vpc_cidr` | CIDR block for VPC | string | n/a |
| `availability_zones` | List of AZs to use | list(string) | n/a |
| `node_groups` | Map of node group configs | map(object) | n/a |
| `kubernetes_version` | Kubernetes version | string | `1.28` |
| `api_public_access` | Enable public API endpoint | bool | `true` |
| `public_access_cidrs` | Allowed CIDRs for public API | list(string) | `["0.0.0.0/0"]` |
| `cluster_log_types` | Control plane log types | list(string) | `[...]` |
| `service_ipv4_cidr` | IPv4 CIDR for services | string | `172.20.0.0/16` |
| `tags` | Common tags | map(string) | `{}` |

## Outputs

- `cluster_name`
- `cluster_endpoint`
- `cluster_certificate`
- `vpc_id`
- `public_subnet_ids`
- `private_subnet_ids`
- `cluster_security_group_id`

## Example

```hcl
module "eks_us_east" {
  source = "./modules/eks"

  region         = "us-east-1"
  cluster_name   = "254carbon-us-east"
  vpc_cidr       = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

  node_groups = {
    general = {
      desired_size   = 3
      min_size       = 2
      max_size       = 10
      instance_types = ["t3.xlarge"]
    }

    ml = {
      desired_size   = 2
      min_size       = 1
      max_size       = 5
      instance_types = ["g4dn.xlarge"]
      capacity_type  = "ON_DEMAND"
    }
  }

  tags = {
    Environment = "production"
    Project     = "254carbon"
  }
}
```
