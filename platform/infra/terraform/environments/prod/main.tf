terraform {
  source = "../.."
}

inputs = {
  environment = "prod"
  aws_region  = "us-east-1"

  # VPC configuration
  vpc_cidr         = "10.2.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets  = ["10.2.1.0/24", "10.2.2.0/24", "10.2.3.0/24"]
  public_subnets   = ["10.2.101.0/24", "10.2.102.0/24", "10.2.103.0/24"]

  # EKS configuration
  node_instance_types = ["t3.large", "t3a.large"]
  node_min_size      = 3
  node_max_size      = 20
  node_desired_size  = 5

  # RDS configuration
  db_instance_class       = "db.t3.large"
  db_allocated_storage   = 100
  db_max_allocated_storage = 1000
  multi_az              = true

  # ElastiCache configuration
  redis_node_type   = "cache.t3.medium"
  redis_num_nodes   = 3
  cluster_mode_enabled = true

  # Monitoring
  pagerduty_integration_key = var.pagerduty_integration_key

  # Domain
  domain_name = "254carbon.com"
}
# Production environment stack
# - High availability and stricter security defaults
# - Larger instance sizes and multi-AZ where applicable
