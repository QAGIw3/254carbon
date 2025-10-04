terraform {
  source = "../.."
}

inputs = {
  environment = "staging"
  aws_region  = "us-east-1"

  # VPC configuration
  vpc_cidr         = "10.1.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets  = ["10.1.1.0/24", "10.1.2.0/24", "10.1.3.0/24"]
  public_subnets   = ["10.1.101.0/24", "10.1.102.0/24", "10.1.103.0/24"]

  # EKS configuration
  node_instance_types = ["t3.large"]
  node_min_size      = 2
  node_max_size      = 6
  node_desired_size  = 3

  # RDS configuration
  db_instance_class       = "db.t3.large"
  db_allocated_storage   = 50
  db_max_allocated_storage = 200
  multi_az              = false

  # ElastiCache configuration
  redis_node_type   = "cache.t3.small"
  redis_num_nodes   = 2

  # Monitoring
  pagerduty_integration_key = var.pagerduty_integration_key  # Set via variable

  # Domain
  domain_name = "254carbon-staging.local"
}
# Staging environment stack
# - Parity with production for integration testing
# - Cost-optimized but representative
