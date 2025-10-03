variable "global_cluster_identifier" {
  description = "Identifier for the Aurora global cluster"
  type        = string
}

variable "engine" {
  description = "Database engine"
  type        = string
  default     = "aurora-postgresql"
}

variable "engine_version" {
  description = "Database engine version"
  type        = string
  default     = "15.3"
}

variable "database_name" {
  description = "Primary database name"
  type        = string
}

variable "primary_region" {
  description = "AWS region for the primary writer cluster"
  type        = string
}

variable "secondary_regions" {
  description = "List of AWS regions for secondary clusters"
  type        = list(string)
  default     = []
}

variable "master_username" {
  description = "Master username for the primary cluster"
  type        = string
}

variable "master_password" {
  description = "Master password for the primary cluster"
  type        = string
  sensitive   = true
}

variable "kms_key_id" {
  description = "KMS key for encryption"
  type        = string
}

variable "instance_class" {
  description = "Instance class for Aurora instances"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "primary_instance_count" {
  description = "Number of writer instances in the primary region"
  type        = number
  default     = 2
}

variable "secondary_instance_count" {
  description = "Number of reader instances per secondary region"
  type        = number
  default     = 1
}

variable "primary_vpc_cidr" {
  description = "CIDR block for the primary region VPC"
  type        = string
}

variable "primary_availability_zones" {
  description = "Availability zones for the primary region"
  type        = list(string)
}

variable "primary_private_subnets" {
  description = "Private subnet CIDRs for the primary VPC"
  type        = list(string)
}

variable "primary_public_subnets" {
  description = "Public subnet CIDRs for the primary VPC"
  type        = list(string)
}

variable "primary_allowed_cidrs" {
  description = "CIDR blocks allowed to connect to the primary cluster"
  type        = list(string)
  default     = []
}

variable "secondary_vpc_cidrs" {
  description = "Map of region to VPC CIDR block for secondary clusters"
  type        = map(string)
  default     = {}
}

variable "secondary_availability_zones" {
  description = "Map of region to list of availability zones"
  type        = map(list(string))
  default     = {}
}

variable "secondary_private_subnets" {
  description = "Map of region to list of private subnet CIDRs"
  type        = map(list(string))
  default     = {}
}

variable "secondary_public_subnets" {
  description = "Map of region to list of public subnet CIDRs"
  type        = map(list(string))
  default     = {}
}

variable "secondary_allowed_cidrs" {
  description = "Map of region to allowed CIDR blocks"
  type        = map(list(string))
  default     = {}
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "preferred_backup_window" {
  description = "Preferred backup window"
  type        = string
  default     = "03:00-05:00"
}

variable "preferred_maintenance_window" {
  description = "Preferred maintenance window"
  type        = string
  default     = "sun:05:00-sun:09:00"
}

variable "parameter_group_family" {
  description = "Parameter group family"
  type        = string
  default     = "aurora-postgresql15"
}

variable "db_timezone" {
  description = "Database timezone"
  type        = string
  default     = "UTC"
}

variable "tags" {
  description = "Common tags applied to all resources"
  type        = map(string)
  default     = {}
}

variable "deletion_protection" {
  description = "Enable deletion protection on clusters"
  type        = bool
  default     = true
}


