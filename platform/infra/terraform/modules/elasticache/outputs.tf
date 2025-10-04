output "cluster_id" {
  description = "ElastiCache cluster ID"
  value       = var.cluster_mode_enabled ? aws_elasticache_replication_group.this[0].id : aws_elasticache_cluster.this.id
}

output "cluster_address" {
  description = "ElastiCache cluster address"
  value       = var.cluster_mode_enabled ? aws_elasticache_replication_group.this[0].configuration_endpoint_address : aws_elasticache_cluster.this.cluster_address
}

output "cluster_port" {
  description = "ElastiCache cluster port"
  value       = 6379
}

output "cluster_configuration_endpoint" {
  description = "ElastiCache cluster configuration endpoint (for cluster mode)"
  value       = var.cluster_mode_enabled ? "${aws_elasticache_replication_group.this[0].configuration_endpoint_address}:${aws_elasticache_replication_group.this[0].port}" : null
}

output "subnet_group_name" {
  description = "ElastiCache subnet group name"
  value       = var.create_subnet_group ? aws_elasticache_subnet_group.this[0].name : var.subnet_group_name
}

output "parameter_group_name" {
  description = "ElastiCache parameter group name"
  value       = var.create_parameter_group ? aws_elasticache_parameter_group.this[0].name : var.parameter_group_name
}

output "auth_token" {
  description = "Auth token for Redis"
  value       = var.auth_token
  sensitive   = true
}

