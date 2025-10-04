variable "cluster_id" {
  description = "ElastiCache cluster ID"
  type        = string
}

variable "engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

variable "parameter_group_name" {
  description = "Parameter group name"
  type        = string
  default     = "default.redis7"
}

variable "create_subnet_group" {
  description = "Create subnet group"
  type        = bool
  default     = true
}

variable "subnet_group_name" {
  description = "Subnet group name"
  type        = string
  default     = ""
}

variable "subnet_group_subnet_ids" {
  description = "Subnet IDs for subnet group"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Security group IDs"
  type        = list(string)
}

variable "at_rest_encryption_enabled" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "transit_encryption_enabled" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

variable "auth_token" {
  description = "Auth token for Redis"
  type        = string
  default     = ""
  sensitive   = true
}

variable "snapshot_window" {
  description = "Snapshot window"
  type        = string
  default     = "05:00-09:00"
}

variable "snapshot_retention_period" {
  description = "Snapshot retention period (days)"
  type        = number
  default     = 0
}

variable "maintenance_window" {
  description = "Maintenance window"
  type        = string
  default     = "wed:03:00-wed:04:00"
}

variable "auto_minor_version_upgrade" {
  description = "Enable auto minor version upgrade"
  type        = bool
  default     = true
}

variable "apply_immediately" {
  description = "Apply changes immediately"
  type        = bool
  default     = false
}

variable "cluster_mode_enabled" {
  description = "Enable cluster mode"
  type        = bool
  default     = false
}

variable "automatic_failover_enabled" {
  description = "Enable automatic failover"
  type        = bool
  default     = false
}

variable "multi_az_enabled" {
  description = "Enable Multi-AZ"
  type        = bool
  default     = false
}

variable "create_parameter_group" {
  description = "Create custom parameter group"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
