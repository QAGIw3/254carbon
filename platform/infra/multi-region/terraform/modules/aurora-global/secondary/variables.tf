variable "region" {
  description = "AWS region for this secondary cluster"
  type        = string
}

variable "global_cluster_identifier" {
  description = "Identifier of the global Aurora cluster"
  type        = string
}

variable "cluster_id_suffix" {
  description = "Suffix for naming secondary resources"
  type        = string
}

variable "engine" {
  description = "Database engine"
  type        = string
}

variable "engine_version" {
  description = "Database engine version"
  type        = string
}

variable "instance_class" {
  description = "Instance class for Aurora instances"
  type        = string
}

variable "instance_count" {
  description = "Number of instances to create"
  type        = number
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

variable "private_subnets" {
  description = "List of private subnet CIDRs"
  type        = list(string)
}

variable "public_subnets" {
  description = "List of public subnet CIDRs"
  type        = list(string)
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to connect"
  type        = list(string)
}

variable "kms_key_id" {
  description = "KMS key for encryption"
  type        = string
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}


