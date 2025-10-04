locals {
  name = var.cluster_name

  # Common tags for all resources
  tags = merge(var.tags, {
    Name = var.cluster_name
  })
}

################################################################################
# EKS Cluster
################################################################################

resource "aws_eks_cluster" "this" {
  name     = var.cluster_name
  version  = var.cluster_version
  role_arn = aws_iam_role.cluster.arn

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = var.cluster_endpoint_public_access
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
  }

  # Enable encryption
  encryption_config {
    provider {
      key_arn = var.cluster_encryption_config.provider_key_arn
    }
    resources = var.cluster_encryption_config.resources
  }

  # Enable control plane logging
  enabled_cluster_log_types = var.cluster_enabled_log_types

  # Ensure that IAM Role permissions are created before and deleted after EKS Cluster handling.
  # Otherwise, EKS will not be able to properly delete EKS managed EC2 infrastructure such as Security Groups.
  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
    aws_iam_role_policy_attachment.cluster_AmazonEKSVPCResourceController,
  ]

  tags = local.tags
}

################################################################################
# EKS Cluster IAM Role
################################################################################

resource "aws_iam_role" "cluster" {
  name = "${var.cluster_name}-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster.name
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSVPCResourceController" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.cluster.name
}

################################################################################
# EKS Node Groups
################################################################################

resource "aws_eks_node_group" "this" {
  for_each = var.node_groups

  cluster_name    = aws_eks_cluster.this.name
  node_group_name = each.value.name
  node_role_arn   = each.value.create_iam_role ? aws_iam_role.node_group[each.key].arn : each.value.iam_role_arn
  subnet_ids      = var.subnet_ids

  ami_type       = each.value.ami_type
  instance_types = each.value.instance_types
  capacity_type  = each.value.capacity_type

  scaling_config {
    desired_size = each.value.desired_size
    max_size     = each.value.max_size
    min_size     = each.value.min_size
  }

  update_config {
    max_unavailable = each.value.max_unavailable
  }

  # Enable IMDSv2 and other security features
  dynamic "taint" {
    for_each = each.value.taints
    content {
      key    = taint.value.key
      value  = taint.value.value
      effect = taint.value.effect
    }
  }

  dynamic "remote_access" {
    for_each = each.value.remote_access != null ? [each.value.remote_access] : []
    content {
      ec2_ssh_key               = remote_access.value.ec2_ssh_key
      source_security_group_ids = remote_access.value.source_security_group_ids
    }
  }

  tags = merge(local.tags, each.value.tags)
}

################################################################################
# EKS Node Group IAM Role
################################################################################

resource "aws_iam_role" "node_group" {
  for_each = { for k, v in var.node_groups : k => v if v.create_iam_role }

  name = "${var.cluster_name}-${each.key}-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "node_group_AmazonEKSWorkerNodePolicy" {
  for_each = { for k, v in var.node_groups : k => v if v.create_iam_role }

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node_group[each.key].name
}

resource "aws_iam_role_policy_attachment" "node_group_AmazonEKS_CNI_Policy" {
  for_each = { for k, v in var.node_groups : k => v if v.create_iam_role }

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node_group[each.key].name
}

resource "aws_iam_role_policy_attachment" "node_group_AmazonEC2ContainerRegistryReadOnly" {
  for_each = { for k, v in var.node_groups : k => v if v.create_iam_role }

  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node_group[each.key].name
}

################################################################################
# Supporting Resources
################################################################################

resource "aws_security_group" "cluster" {
  count = var.create_cluster_security_group ? 1 : 0

  name_prefix = "${var.cluster_name}-cluster-"
  vpc_id      = var.vpc_id

  ingress {
    description = "Cluster API access"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.this.cidr_block]
  }

  tags = merge(local.tags, {
    Name = "${var.cluster_name}-cluster"
  })
}

data "aws_vpc" "this" {
  id = var.vpc_id
}

################################################################################
# KMS Key for Cluster Encryption
################################################################################

resource "aws_kms_key" "cluster" {
  count = var.create_kms_key ? 1 : 0

  description             = "EKS cluster encryption key"
  deletion_window_in_days = var.kms_key_deletion_window_in_days

  tags = local.tags
}

resource "aws_kms_alias" "cluster" {
  count = var.create_kms_key ? 1 : 0

  name          = "alias/${var.cluster_name}-cluster"
  target_key_id = aws_kms_key.cluster[0].key_id
}
