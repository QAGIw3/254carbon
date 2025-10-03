############################################################
# 254Carbon Terraform Module: Aurora Global Database
#
# Provisions an Amazon Aurora Global Database with a primary
# cluster in one region and secondary read replicas in
# additional regions for disaster recovery and low-latency
# reads.
############################################################

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

locals {
  primary_region    = var.primary_region
  secondary_regions = var.secondary_regions
  cluster_id        = var.global_cluster_identifier
}

################################
# Global Cluster Definition    #
################################

resource "aws_rds_global_cluster" "this" {
  global_cluster_identifier = local.cluster_id
  engine                    = var.engine
  engine_version            = var.engine_version
  database_name             = var.database_name
  storage_encrypted         = true
  deletion_protection       = var.deletion_protection
}

################################
# Primary Cluster (Writer)     #
################################

provider "aws" {
  alias  = "primary"
  region = local.primary_region
}

module "primary_vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  providers = {
    aws = aws.primary
  }

  name = "${local.cluster_id}-primary"
  cidr = var.primary_vpc_cidr

  azs             = var.primary_availability_zones
  private_subnets = var.primary_private_subnets
  public_subnets  = var.primary_public_subnets

  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true
  enable_ipv6          = false

  tags = var.tags
}

resource "aws_rds_cluster_parameter_group" "primary" {
  provider = aws.primary

  name        = "${local.cluster_id}-primary-pg"
  family      = var.parameter_group_family
  description = "Primary cluster parameter group"

  parameter {
    name  = "timezone"
    value = var.db_timezone
  }
}

resource "aws_security_group" "primary" {
  provider = aws.primary

  name        = "${local.cluster_id}-primary-sg"
  description = "Security group for Aurora primary cluster"
  vpc_id      = module.primary_vpc.vpc_id

  ingress {
    description = "PostgreSQL ingress"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.primary_allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_db_subnet_group" "primary" {
  provider = aws.primary

  name       = "${local.cluster_id}-primary-subnet"
  subnet_ids = module.primary_vpc.private_subnets

  tags = var.tags
}

resource "aws_rds_cluster" "primary" {
  provider = aws.primary

  cluster_identifier   = "${local.cluster_id}-primary"
  engine               = var.engine
  engine_version       = var.engine_version
  global_cluster_identifier = aws_rds_global_cluster.this.id
  database_name        = var.database_name
  master_username      = var.master_username
  master_password      = var.master_password
  storage_encrypted    = true
  kms_key_id           = var.kms_key_id
  backup_retention_period = var.backup_retention_days
  preferred_backup_window  = var.preferred_backup_window
  preferred_maintenance_window = var.preferred_maintenance_window
  deletion_protection   = var.deletion_protection
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.primary.name
  db_subnet_group_name  = aws_db_subnet_group.primary.name
  vpc_security_group_ids = [aws_security_group.primary.id]
  port                  = 5432

  copy_tags_to_snapshot = true

  tags = var.tags
}

resource "aws_rds_cluster_instance" "primary" {
  provider = aws.primary

  count = var.primary_instance_count

  cluster_identifier = aws_rds_cluster.primary.id
  identifier         = "${local.cluster_id}-primary-${count.index + 1}"
  instance_class     = var.instance_class
  engine             = var.engine
  engine_version     = var.engine_version
  publicly_accessible = false
  promotion_tier      = 1

  auto_minor_version_upgrade = true

  tags = var.tags
}

################################
# Secondary Clusters (Readers) #
################################

module "secondary" {
  source = "./secondary"

  for_each = toset(local.secondary_regions)

  providers = {
    aws = aws
  }

  region                      = each.value
  global_cluster_identifier    = aws_rds_global_cluster.this.id
  cluster_id_suffix            = each.value
  engine                       = var.engine
  engine_version               = var.engine_version
  instance_class               = var.instance_class
  instance_count               = var.secondary_instance_count
  vpc_cidr                     = lookup(var.secondary_vpc_cidrs, each.value, "10.10.0.0/16")
  availability_zones           = lookup(var.secondary_availability_zones, each.value, [])
  private_subnets              = lookup(var.secondary_private_subnets, each.value, [])
  public_subnets               = lookup(var.secondary_public_subnets, each.value, [])
  allowed_cidrs                = lookup(var.secondary_allowed_cidrs, each.value, ["10.0.0.0/8"])
  kms_key_id                   = var.kms_key_id
  tags                         = var.tags
}

################################
# Outputs                      #
################################

output "primary_endpoint" {
  value = aws_rds_cluster.primary.endpoint
}

output "reader_endpoint" {
  value = aws_rds_cluster.primary.reader_endpoint
}

output "arn" {
  value = aws_rds_global_cluster.this.arn
}

output "cluster_id" {
  value = aws_rds_global_cluster.this.id
}


