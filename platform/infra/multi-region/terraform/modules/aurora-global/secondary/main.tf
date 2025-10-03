terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  alias  = "this"
  region = var.region
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  providers = {
    aws = aws.this
  }

  name = "${var.global_cluster_identifier}-${var.cluster_id_suffix}"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true
  enable_ipv6          = false

  tags = var.tags
}

resource "aws_security_group" "secondary" {
  provider = aws.this

  name        = "${var.global_cluster_identifier}-${var.cluster_id_suffix}-sg"
  description = "Security group for Aurora secondary cluster"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "PostgreSQL ingress"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_db_subnet_group" "secondary" {
  provider = aws.this

  name       = "${var.global_cluster_identifier}-${var.cluster_id_suffix}-subnet"
  subnet_ids = module.vpc.private_subnets

  tags = var.tags
}

resource "aws_rds_cluster" "secondary" {
  provider = aws.this

  cluster_identifier          = "${var.global_cluster_identifier}-${var.cluster_id_suffix}"
  engine                      = var.engine
  engine_version              = var.engine_version
  global_cluster_identifier   = var.global_cluster_identifier
  storage_encrypted           = true
  kms_key_id                  = var.kms_key_id
  db_subnet_group_name        = aws_db_subnet_group.secondary.name
  vpc_security_group_ids      = [aws_security_group.secondary.id]
  deletion_protection         = true
  skip_final_snapshot         = false
  copy_tags_to_snapshot       = true

  tags = var.tags
}

resource "aws_rds_cluster_instance" "secondary" {
  provider = aws.this

  count               = var.instance_count
  cluster_identifier   = aws_rds_cluster.secondary.id
  identifier           = "${var.global_cluster_identifier}-${var.cluster_id_suffix}-${count.index + 1}"
  instance_class       = var.instance_class
  engine               = var.engine
  engine_version       = var.engine_version
  publicly_accessible  = false
  auto_minor_version_upgrade = true
  promotion_tier       = 2

  tags = var.tags
}

output "reader_endpoint" {
  value = aws_rds_cluster.secondary.reader_endpoint
}


