output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.this.name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.this.endpoint
}

output "cluster_certificate_authority_data" {
  description = "EKS cluster certificate authority data"
  value       = aws_eks_cluster.this.certificate_authority[0].data
}

output "oidc_provider_arn" {
  description = "EKS OIDC provider ARN"
  value       = aws_eks_cluster.this.identity[0].oidc[0].issuer
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = var.create_cluster_security_group ? aws_security_group.cluster[0].id : null
}

output "node_group_role_arn" {
  description = "EKS node group IAM role ARN"
  value       = { for k, v in aws_iam_role.node_group : k => v.arn }
}

output "kubeconfig" {
  description = "Kubernetes config for accessing the cluster"
  value = templatefile("${path.module}/templates/kubeconfig.tpl", {
    cluster_name                 = aws_eks_cluster.this.name
    cluster_endpoint             = aws_eks_cluster.this.endpoint
    cluster_certificate_authority_data = aws_eks_cluster.this.certificate_authority[0].data
  })
  sensitive = true
}

