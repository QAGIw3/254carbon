variable "name" {
  description = "Name of the VPC"
  type        = string
}

variable "cidr_block" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateway"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use single NAT gateway for all AZs"
  type        = bool
  default     = false
}

variable "create_igw" {
  description = "Create internet gateway"
  type        = bool
  default     = true
}

variable "create_public_route_table" {
  description = "Create public route table"
  type        = bool
  default     = true
}

variable "create_private_route_table" {
  description = "Create private route table"
  type        = bool
  default     = true
}

variable "create_database_subnet_group" {
  description = "Create database subnet group"
  type        = bool
  default     = true
}

variable "create_elasticache_subnet_group" {
  description = "Create ElastiCache subnet group"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
