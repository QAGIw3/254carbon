terraform {
  source = "../.."
}

inputs = {
  environment = "dev"
  aws_region  = "us-east-1"

  # VPC configuration
  vpc_cidr         = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b"]
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets   = ["10.0.101.0/24", "10.0.102.0/24"]

  # EKS configuration
  node_instance_types = ["t3.medium"]
  node_min_size      = 1
  node_max_size      = 3
  node_desired_size  = 2

  # RDS configuration
  db_instance_class       = "db.t3.medium"
  db_allocated_storage   = 20
  db_max_allocated_storage = 50

  # ElastiCache configuration
  redis_node_type   = "cache.t3.micro"
  redis_num_nodes   = 1

  # Monitoring
  pagerduty_integration_key = ""  # No alerting for dev

  # Domain
  domain_name = "254carbon-dev.local"
}
# Development environment stack
# - Lower resource sizes and costs
# - Enables rapid iteration and sandboxing
