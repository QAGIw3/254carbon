output "vpc_id" {
  description = "The ID of the VPC"
  value       = aws_vpc.this.id
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = aws_vpc.this.cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = aws_subnet.private[*].id
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "database_subnet_group" {
  description = "ID of database subnet group"
  value       = var.create_database_subnet_group ? aws_db_subnet_group.database[0].name : null
}

output "elasticache_subnet_group" {
  description = "ID of ElastiCache subnet group"
  value       = var.create_elasticache_subnet_group ? aws_elasticache_subnet_group.this[0].name : null
}

output "rds_security_group_id" {
  description = "ID of the RDS security group"
  value       = aws_security_group.rds.id
}

output "redis_security_group_id" {
  description = "ID of the Redis security group"
  value       = aws_security_group.redis.id
}

output "eks_security_group_id" {
  description = "ID of the EKS security group"
  value       = aws_security_group.eks.id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = var.create_igw ? aws_internet_gateway.this[0].id : null
}

output "nat_gateway_ids" {
  description = "List of NAT Gateway IDs"
  value       = aws_nat_gateway.this[*].id
}
