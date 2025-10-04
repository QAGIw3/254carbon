output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "eks_cluster_ca_certificate" {
  description = "EKS cluster CA certificate"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "eks_oidc_provider_arn" {
  description = "EKS OIDC provider ARN"
  value       = module.eks.oidc_provider_arn
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
  sensitive   = true
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.db_instance_name
}

output "rds_username" {
  description = "RDS master username"
  value       = module.rds.db_instance_username
  sensitive   = true
}

output "rds_password" {
  description = "RDS master password"
  value       = module.rds.db_instance_password
  sensitive   = true
}

output "elasticache_cluster_endpoint" {
  description = "ElastiCache cluster endpoint"
  value       = module.elasticache.cluster_address
  sensitive   = true
}

output "elasticache_port" {
  description = "ElastiCache port"
  value       = module.elasticache.cluster_port
}

output "grafana_url" {
  description = "Grafana URL"
  value       = module.monitoring.grafana_url
}

output "grafana_admin_password" {
  description = "Grafana admin password"
  value       = module.monitoring.grafana_admin_password
  sensitive   = true
}

output "kubeconfig" {
  description = "Kubernetes config for accessing the cluster"
  value       = module.eks.kubeconfig
  sensitive   = true
}

