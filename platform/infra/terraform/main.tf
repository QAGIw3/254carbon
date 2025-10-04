terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket         = "254carbon-terraform-state"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "254carbon-terraform-locks"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment   = var.environment
      Project       = "254Carbon"
      ManagedBy     = "Terraform"
      CostCenter    = "Platform"
    }
  }
}

module "vpc" {
  source = "./modules/vpc"

  name                 = "254carbon-${var.environment}"
  cidr_block          = var.vpc_cidr
  availability_zones  = var.availability_zones
  private_subnets     = var.private_subnets
  public_subnets      = var.public_subnets
  enable_nat_gateway  = true
  single_nat_gateway  = var.environment != "prod" # Use single NAT for non-prod to save costs

  tags = {
    Environment = var.environment
  }
}

module "eks" {
  source = "./modules/eks"

  cluster_name    = "254carbon-${var.environment}"
  cluster_version = "1.27"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  # Node groups configuration
  node_groups = {
    default = {
      name          = "default"
      instance_types = var.node_instance_types
      min_size      = var.node_min_size
      max_size      = var.node_max_size
      desired_size  = var.node_desired_size

      # Enable IMDSv2 and other security features
      enable_bootstrap_user_data = true
      bootstrap_extra_args      = "--kubelet-extra-args '--node-labels=node.kubernetes.io/lifecycle=spot --register-with-taints=spot=true:NoSchedule'"

      # Security configurations
      create_security_group = true
      create_iam_role       = true

      tags = {
        Environment = var.environment
      }
    }
  }

  # Enable encryption and logging
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  cluster_enabled_log_types = [
    "api", "audit", "authenticator", "controllerManager", "scheduler"
  ]

  tags = {
    Environment = var.environment
  }
}

module "rds" {
  source = "./modules/rds"

  identifier = "254carbon-${var.environment}"

  # Database configuration
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.db_instance_class

  # Storage configuration
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn

  # High availability
  multi_az               = var.environment == "prod"
  backup_retention_period = var.environment == "prod" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  # Security
  vpc_security_group_ids = [module.vpc.rds_security_group_id]
  db_subnet_group_name   = module.vpc.database_subnet_group

  # Performance and monitoring
  performance_insights_enabled    = true
  performance_insights_kms_key_id = aws_kms_key.rds.arn
  monitoring_interval             = var.environment == "prod" ? 30 : 60
  monitoring_role_arn             = aws_iam_role.rds_enhanced_monitoring.arn

  # Backup and recovery
  deletion_protection = var.environment == "prod"
  skip_final_snapshot = var.environment != "prod"

  tags = {
    Environment = var.environment
  }
}

module "elasticache" {
  source = "./modules/elasticache"

  cluster_id         = "254carbon-${var.environment}"
  engine_version     = "7.0"
  node_type          = var.redis_node_type
  num_cache_nodes    = var.redis_num_nodes
  parameter_group_name = "default.redis7"

  # High availability
  automatic_failover_enabled = var.environment == "prod"
  multi_az_enabled          = var.environment == "prod"

  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = random_password.redis_auth.result

  # Performance
  snapshot_retention_period = var.environment == "prod" ? 7 : 0
  snapshot_window          = "05:00-09:00"

  # Network
  subnet_group_name = module.vpc.elasticache_subnet_group
  security_group_ids = [module.vpc.redis_security_group_id]

  tags = {
    Environment = var.environment
  }
}

module "monitoring" {
  source = "./modules/monitoring"

  cluster_name = module.eks.cluster_name
  eks_oidc_provider_arn = module.eks.oidc_provider_arn

  # Grafana configuration
  grafana_admin_password = random_password.grafana_admin.result

  # Alerting
  pagerduty_integration_key = var.pagerduty_integration_key

  tags = {
    Environment = var.environment
  }
}

# KMS keys for encryption
resource "aws_kms_key" "eks" {
  description             = "EKS cluster encryption key"
  deletion_window_in_days = var.environment == "prod" ? 30 : 7

  tags = {
    Environment = var.environment
  }
}

resource "aws_kms_key" "rds" {
  description             = "RDS encryption key"
  deletion_window_in_days = var.environment == "prod" ? 30 : 7

  tags = {
    Environment = var.environment
  }
}

# IAM roles for monitoring
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "254carbon-rds-enhanced-monitoring-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Random passwords for services
resource "random_password" "redis_auth" {
  length  = 32
  special = true
}

resource "random_password" "grafana_admin" {
  length  = 16
  special = true
}
