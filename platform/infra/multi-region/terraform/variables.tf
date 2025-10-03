variable "db_master_username" {
  description = "Master username for Aurora"
  type        = string
}

variable "db_master_password" {
  description = "Master password for Aurora"
  type        = string
  sensitive   = true
}

variable "db_kms_key_arn" {
  description = "KMS key ARN used for database encryption"
  type        = string
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for CloudFront"
  type        = string
}

variable "provider_role_arn" {
  description = "IAM role ARN assumed by Terraform when managing AWS resources"
  type        = string
}

variable "root_domain" {
  description = "Root domain managed by Route53"
  type        = string
  default     = "254carbon.ai"
}

variable "api_subdomain" {
  description = "API subdomain prefix"
  type        = string
  default     = "api"
}

variable "regional_api_endpoints" {
  description = "Map of region to API endpoint settings (domain and hosted zone id)"
  type = map(object({
    domain_name    = string
    hosted_zone_id = string
  }))
}

variable "redis_node_type" {
  description = "ElastiCache node instance type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.1"
}

variable "redis_replicas_per_node_group" {
  description = "Number of replicas per Redis node group"
  type        = number
  default     = 1
}

variable "redis_parameter_group_name" {
  description = "Parameter group for Redis replication groups"
  type        = string
  default     = "default.redis7"
}
