locals {
  name = var.identifier

  # Common tags for all resources
  tags = merge(var.tags, {
    Name = var.identifier
  })
}

################################################################################
# RDS Instance
################################################################################

resource "aws_db_instance" "this" {
  identifier = var.identifier

  # Engine configuration
  engine         = var.engine
  engine_version = var.engine_version
  instance_class = var.instance_class

  # Database configuration
  db_name  = var.database_name
  username = var.username != "" ? var.username : "postgres"
  password = var.password != "" ? var.password : random_password.db_password.result

  # Storage configuration
  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = var.storage_type
  storage_encrypted     = var.storage_encrypted
  kms_key_id           = var.kms_key_id

  # Network configuration
  db_subnet_group_name   = var.db_subnet_group_name
  vpc_security_group_ids = var.vpc_security_group_ids

  # High availability and backup
  multi_az               = var.multi_az
  backup_retention_period = var.backup_retention_period
  backup_window          = var.backup_window
  maintenance_window     = var.maintenance_window

  # Performance and monitoring
  performance_insights_enabled    = var.performance_insights_enabled
  performance_insights_kms_key_id = var.performance_insights_kms_key_id
  monitoring_interval             = var.monitoring_interval
  monitoring_role_arn             = var.monitoring_role_arn

  # Security and compliance
  deletion_protection = var.deletion_protection
  skip_final_snapshot = var.skip_final_snapshot
  final_snapshot_identifier = var.skip_final_snapshot ? null : "${var.identifier}-final-snapshot"

  # Auto minor version upgrade
  auto_minor_version_upgrade = var.auto_minor_version_upgrade

  # Additional configuration
  apply_immediately = var.apply_immediately

  tags = local.tags
}

################################################################################
# Random password for database
################################################################################

resource "random_password" "db_password" {
  count = var.password == "" ? 1 : 0

  length  = 32
  special = true
}

################################################################################
# RDS Proxy (for production high availability)
################################################################################

resource "aws_db_proxy" "this" {
  count = var.create_proxy ? 1 : 0

  name                   = "${var.identifier}-proxy"
  engine_family          = var.engine == "postgres" ? "POSTGRESQL" : "MYSQL"
  idle_client_timeout    = 1800
  require_tls           = true
  role_arn              = aws_iam_role.rds_proxy[0].arn
  vpc_subnet_ids        = var.vpc_subnet_ids
  vpc_security_group_ids = var.vpc_security_group_ids

  auth {
    auth_scheme = "SECRETS"
    secret_arn  = aws_secretsmanager_secret.rds_proxy[0].arn
  }

  tags = local.tags
}

resource "aws_iam_role" "rds_proxy" {
  count = var.create_proxy ? 1 : 0

  name = "${var.identifier}-proxy-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "rds_proxy" {
  count = var.create_proxy ? 1 : 0

  role       = aws_iam_role.rds_proxy[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSProxyRolePolicy"
}

resource "aws_secretsmanager_secret" "rds_proxy" {
  count = var.create_proxy ? 1 : 0

  name = "${var.identifier}-proxy-secret"

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "rds_proxy" {
  count = var.create_proxy ? 1 : 0

  secret_id = aws_secretsmanager_secret.rds_proxy[0].id
  secret_string = jsonencode({
    username = var.username != "" ? var.username : "postgres"
    password = var.password != "" ? var.password : random_password.db_password[0].result
  })
}

################################################################################
# Enhanced monitoring IAM role
################################################################################

resource "aws_iam_role" "rds_enhanced_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0

  name = "${var.identifier}-enhanced-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0

  role       = aws_iam_role.rds_enhanced_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}
