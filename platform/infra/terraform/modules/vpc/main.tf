locals {
  name = var.name

  # Common tags for all resources
  tags = merge(var.tags, {
    Name = var.name
  })
}

################################################################################
# VPC
################################################################################

resource "aws_vpc" "this" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = local.tags
}

################################################################################
# Internet Gateway
################################################################################

resource "aws_internet_gateway" "this" {
  count = var.create_igw ? 1 : 0

  vpc_id = aws_vpc.this.id

  tags = merge(local.tags, {
    Name = "${local.name}-igw"
  })
}

################################################################################
# Public Subnets
################################################################################

resource "aws_subnet" "public" {
  count = length(var.public_subnets)

  vpc_id                  = aws_vpc.this.id
  cidr_block              = element(var.public_subnets, count.index)
  availability_zone       = element(var.availability_zones, count.index)
  map_public_ip_on_launch = true

  tags = merge(local.tags, {
    Name = "${local.name}-public-${element(var.availability_zones, count.index)}"
    Type = "Public"
  })
}

################################################################################
# Private Subnets
################################################################################

resource "aws_subnet" "private" {
  count = length(var.private_subnets)

  vpc_id            = aws_vpc.this.id
  cidr_block        = element(var.private_subnets, count.index)
  availability_zone = element(var.availability_zones, count.index)

  tags = merge(local.tags, {
    Name = "${local.name}-private-${element(var.availability_zones, count.index)}"
    Type = "Private"
  })
}

resource "aws_db_subnet_group" "database" {
  count = var.create_database_subnet_group ? 1 : 0

  name       = "${local.name}-database"
  subnet_ids = aws_subnet.private[*].id

  tags = local.tags
}

resource "aws_elasticache_subnet_group" "this" {
  count = var.create_elasticache_subnet_group ? 1 : 0

  name       = "${local.name}-elasticache"
  subnet_ids = aws_subnet.private[*].id

  tags = local.tags
}

################################################################################
# NAT Gateways
################################################################################

resource "aws_eip" "nat" {
  count = var.enable_nat_gateway && var.single_nat_gateway == false ? length(var.availability_zones) : var.enable_nat_gateway ? 1 : 0

  domain = "vpc"
  depends_on = [aws_internet_gateway.this]

  tags = merge(local.tags, {
    Name = "${local.name}-nat-${count.index + 1}"
  })
}

resource "aws_nat_gateway" "this" {
  count = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(var.availability_zones)) : 0

  allocation_id = element(aws_eip.nat[*].id, count.index)
  subnet_id     = element(aws_subnet.public[*].id, count.index)

  tags = merge(local.tags, {
    Name = "${local.name}-nat-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.this]
}

################################################################################
# Route Tables
################################################################################

resource "aws_route_table" "public" {
  count = var.create_public_route_table ? 1 : 0

  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this[0].id
  }

  tags = merge(local.tags, {
    Name = "${local.name}-public"
  })
}

resource "aws_route_table" "private" {
  count = var.create_private_route_table ? length(var.availability_zones) : 0

  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = element(aws_nat_gateway.this[*].id, count.index)
  }

  tags = merge(local.tags, {
    Name = "${local.name}-private-${element(var.availability_zones, count.index)}"
  })
}

resource "aws_route_table_association" "public" {
  count = length(var.public_subnets)

  subnet_id      = element(aws_subnet.public[*].id, count.index)
  route_table_id = aws_route_table.public[0].id
}

resource "aws_route_table_association" "private" {
  count = length(var.private_subnets)

  subnet_id      = element(aws_subnet.private[*].id, count.index)
  route_table_id = element(aws_route_table.private[*].id, count.index)
}

################################################################################
# Security Groups
################################################################################

resource "aws_security_group" "rds" {
  name_prefix = "${local.name}-rds-"
  vpc_id      = aws_vpc.this.id

  ingress {
    description = "PostgreSQL"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.this.cidr_block]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-rds"
  })
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name}-redis-"
  vpc_id      = aws_vpc.this.id

  ingress {
    description = "Redis"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.this.cidr_block]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-redis"
  })
}

resource "aws_security_group" "eks" {
  name_prefix = "${local.name}-eks-"
  vpc_id      = aws_vpc.this.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-eks"
  })
}
