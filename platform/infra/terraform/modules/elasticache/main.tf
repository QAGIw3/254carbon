locals {
  name = var.cluster_id

  # Common tags for all resources
  tags = merge(var.tags, {
    Name = var.cluster_id
  })
}

################################################################################
# ElastiCache Subnet Group
################################################################################

resource "aws_elasticache_subnet_group" "this" {
  count = var.create_subnet_group ? 1 : 0

  name       = "${var.cluster_id}-subnet-group"
  subnet_ids = var.subnet_group_subnet_ids

  tags = local.tags
}

################################################################################
# ElastiCache Cluster
################################################################################

resource "aws_elasticache_cluster" "this" {
  cluster_id           = var.cluster_id
  engine              = "redis"
  engine_version      = var.engine_version
  node_type           = var.node_type
  num_cache_nodes     = var.num_cache_nodes
  parameter_group_name = var.parameter_group_name

  # Network configuration
  subnet_group_name  = var.create_subnet_group ? aws_elasticache_subnet_group.this[0].name : var.subnet_group_name
  security_group_ids = var.security_group_ids

  # Security configuration
  at_rest_encryption_enabled = var.at_rest_encryption_enabled
  transit_encryption_enabled = var.transit_encryption_enabled
  auth_token                = var.auth_token

  # Maintenance and backup
  snapshot_window          = var.snapshot_window
  snapshot_retention_period = var.snapshot_retention_period
  maintenance_window       = var.maintenance_window

  # Auto minor version upgrade
  auto_minor_version_upgrade = var.auto_minor_version_upgrade

  # Additional configuration
  apply_immediately = var.apply_immediately

  tags = local.tags
}

################################################################################
# ElastiCache Replication Group (for cluster mode)
################################################################################

resource "aws_elasticache_replication_group" "this" {
  count = var.cluster_mode_enabled ? 1 : 0

  replication_group_id         = var.cluster_id
  description                 = "254Carbon Redis cluster"
  engine_version              = var.engine_version
  node_type                   = var.node_type
  port                        = 6379
  parameter_group_name        = var.parameter_group_name

  # Cluster configuration
  num_cache_clusters          = var.num_cache_nodes
  automatic_failover_enabled  = var.automatic_failover_enabled
  multi_az_enabled           = var.multi_az_enabled

  # Network configuration
  subnet_group_name  = var.create_subnet_group ? aws_elasticache_subnet_group.this[0].name : var.subnet_group_name
  security_group_ids = var.security_group_ids

  # Security configuration
  at_rest_encryption_enabled = var.at_rest_encryption_enabled
  transit_encryption_enabled = var.transit_encryption_enabled
  auth_token                = var.auth_token

  # Maintenance and backup
  snapshot_window          = var.snapshot_window
  snapshot_retention_period = var.snapshot_retention_period
  maintenance_window       = var.maintenance_window

  # Auto minor version upgrade
  auto_minor_version_upgrade = var.auto_minor_version_upgrade

  # Additional configuration
  apply_immediately = var.apply_immediately

  tags = local.tags
}

################################################################################
# ElastiCache Parameter Group
################################################################################

resource "aws_elasticache_parameter_group" "this" {
  count = var.create_parameter_group ? 1 : 0

  family = "redis7"
  name   = "${var.cluster_id}-params"

  # Performance parameters
  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  tags = local.tags
}

