output "grafana_url" {
  description = "Grafana URL"
  value       = "https://${var.domain}/grafana"
}

output "grafana_admin_password" {
  description = "Grafana admin password"
  value       = var.grafana_admin_password
  sensitive   = true
}

output "prometheus_url" {
  description = "Prometheus URL"
  value       = "https://${var.domain}/prometheus"
}

output "alertmanager_url" {
  description = "AlertManager URL"
  value       = "https://${var.domain}/alertmanager"
}

output "monitoring_namespace" {
  description = "Monitoring namespace"
  value       = kubernetes_namespace.monitoring.metadata[0].name
}

