variable "region" {
  description = "AWS region for the EKS cluster"
  type        = string
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
}

variable "node_groups" {
  description = "Map of managed node group configurations"
  type = map(object({
    desired_size  = number
    min_size      = number
    max_size      = number
    instance_types = list(string)
    capacity_type  = optional(string)
    ami_type       = optional(string)
    labels         = optional(map(string))
    taints         = optional(list(object({
      key    = string
      value  = string
      effect = string
    })))
    max_unavailable = optional(number)
  }))
}

variable "kubernetes_version" {
  description = "Desired Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "api_public_access" {
  description = "Whether the API server endpoint is publicly accessible"
  type        = bool
  default     = true
}

variable "public_access_cidrs" {
  description = "List of CIDR blocks allowed to access the public API endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "cluster_log_types" {
  description = "List of control plane logs to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

variable "service_ipv4_cidr" {
  description = "IPv4 CIDR for Kubernetes services"
  type        = string
  default     = "172.20.0.0/16"
}

variable "tags" {
  description = "Common tags applied to all resources"
  type        = map(string)
  default     = {}
}


