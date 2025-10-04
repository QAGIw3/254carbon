output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.this.id
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.this.arn
}

output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.this.endpoint
}

output "db_instance_address" {
  description = "RDS instance address"
  value       = aws_db_instance.this.address
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = aws_db_instance.this.port
}

output "db_instance_name" {
  description = "RDS database name"
  value       = aws_db_instance.this.db_name
}

output "db_instance_username" {
  description = "RDS master username"
  value       = aws_db_instance.this.username
}

output "db_instance_password" {
  description = "RDS master password"
  value       = var.password != "" ? var.password : random_password.db_password[0].result
  sensitive   = true
}

output "db_proxy_endpoint" {
  description = "RDS proxy endpoint"
  value       = var.create_proxy ? aws_db_proxy.this[0].endpoint : null
}

output "db_proxy_target_group_arn" {
  description = "RDS proxy target group ARN"
  value       = var.create_proxy ? aws_db_proxy.this[0].target_group_arn : null
}

output "enhanced_monitoring_role_arn" {
  description = "Enhanced monitoring IAM role ARN"
  value       = var.monitoring_interval > 0 ? aws_iam_role.rds_enhanced_monitoring[0].arn : null
}
