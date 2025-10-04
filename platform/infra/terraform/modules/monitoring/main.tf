locals {
  name = "254carbon-${var.environment}"

  # Common tags for all resources
  tags = merge(var.tags, {
    Name = local.name
  })
}

################################################################################
# Grafana
################################################################################

resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  version    = "6.58.0"
  namespace  = "monitoring"

  create_namespace = true

  values = [
    templatefile("${path.module}/templates/grafana-values.yaml", {
      admin_password = var.grafana_admin_password
      domain        = var.domain
    })
  ]

  depends_on = [time_sleep.wait_for_namespace]
}

################################################################################
# Prometheus
################################################################################

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "48.1.1"
  namespace  = "monitoring"

  create_namespace = true

  values = [
    templatefile("${path.module}/templates/prometheus-values.yaml", {
      cluster_name = var.cluster_name
    })
  ]

  depends_on = [time_sleep.wait_for_namespace]
}

################################################################################
# AlertManager Configuration
################################################################################

resource "kubectl_manifest" "alertmanager_config" {
  yaml_body = templatefile("${path.module}/templates/alertmanager-config.yaml", {
    pagerduty_integration_key = var.pagerduty_integration_key
  })

  depends_on = [helm_release.prometheus]
}

################################################################################
# Monitoring Namespace
################################################################################

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"

    labels = {
      name = "monitoring"
    }
  }
}

# Ensure namespace exists before deploying monitoring stack
resource "time_sleep" "wait_for_namespace" {
  depends_on = [kubernetes_namespace.monitoring]

  create_duration = "10s"
}

################################################################################
# Grafana Dashboards ConfigMap
################################################################################

resource "kubectl_manifest" "grafana_dashboards" {
  yaml_body = templatefile("${path.module}/templates/grafana-dashboards.yaml", {})

  depends_on = [helm_release.grafana]
}

################################################################################
# Business KPI Dashboard ConfigMap
################################################################################

resource "kubernetes_config_map" "business_kpi_dashboard" {
  metadata {
    name      = "254carbon-business-kpis-dashboard"
    namespace = "monitoring"
    labels = {
      grafana_dashboard = "1"
    }
  }

  data = {
    "254carbon-business-kpis.json" = file("${path.module}/templates/business-kpis-dashboard.json")
  }

  depends_on = [helm_release.grafana]
}

################################################################################
# Prometheus Rules for 254Carbon
################################################################################

resource "kubectl_manifest" "prometheus_rules" {
  yaml_body = templatefile("${path.module}/templates/prometheus-rules.yaml", {})

  depends_on = [helm_release.prometheus]
}
