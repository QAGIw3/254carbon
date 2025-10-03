############################################################
# 254Carbon Terraform Module: Amazon EKS Cluster
#
# Provisions Kubernetes control plane, managed node groups,
# supporting IAM roles/policies, and network dependencies.
# Designed for multi-region production clusters with GPU
# workloads support (optional).
############################################################

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }
}

locals {
  eks_name = var.cluster_name

  az_count = length(var.availability_zones)

  public_subnet_cidrs = [
    for idx in range(local.az_count) : cidrsubnet(var.vpc_cidr, 4, idx)
  ]

  private_subnet_cidrs = [
    for idx in range(local.az_count) : cidrsubnet(var.vpc_cidr, 4, idx + 10)
  ]

  az_map = {
    for idx, az in var.availability_zones : idx => az
  }
}

####################
# Networking Layer #
####################

resource "aws_vpc" "eks" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(var.tags, {
    Name = "${local.eks_name}-vpc"
  })
}

resource "aws_internet_gateway" "eks" {
  vpc_id = aws_vpc.eks.id

  tags = merge(var.tags, {
    Name = "${local.eks_name}-igw"
  })
}

resource "aws_subnet" "public" {
  for_each = local.az_map

  vpc_id                  = aws_vpc.eks.id
  cidr_block              = local.public_subnet_cidrs[each.key]
  map_public_ip_on_launch = true
  availability_zone       = each.value

  tags = merge(var.tags, {
    Name                           = "${local.eks_name}-public-${each.key}"
    "kubernetes.io/cluster/${local.eks_name}" = "shared"
    "kubernetes.io/role/elb"           = "1"
  })
}

resource "aws_subnet" "private" {
  for_each = local.az_map

  vpc_id            = aws_vpc.eks.id
  cidr_block        = local.private_subnet_cidrs[each.key]
  availability_zone = each.value

  tags = merge(var.tags, {
    Name                           = "${local.eks_name}-private-${each.key}"
    "kubernetes.io/cluster/${local.eks_name}" = "shared"
    "kubernetes.io/role/internal-elb"     = "1"
  })
}

resource "aws_eip" "nat" {
  for_each = aws_subnet.public

  vpc = true

  tags = merge(var.tags, {
    Name = "${local.eks_name}-nat-${each.key}"
  })
}

resource "aws_nat_gateway" "eks" {
  for_each = aws_subnet.public

  subnet_id     = each.value.id
  allocation_id = aws_eip.nat[each.key].id

  tags = merge(var.tags, {
    Name = "${local.eks_name}-nat-${each.key}"
  })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.eks.id

  tags = merge(var.tags, {
    Name = "${local.eks_name}-public"
  })
}

resource "aws_route" "public_internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.eks.id
}

resource "aws_route_table_association" "public" {
  for_each = aws_subnet.public

  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  for_each = aws_nat_gateway.eks

  vpc_id = aws_vpc.eks.id

  tags = merge(var.tags, {
    Name = "${local.eks_name}-private-${each.key}"
  })
}

resource "aws_route" "private_nat" {
  for_each = aws_route_table.private

  route_table_id         = each.value.id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.eks[each.key].id
}

resource "aws_route_table_association" "private" {
  for_each = aws_subnet.private

  subnet_id      = each.value.id
  route_table_id = aws_route_table.private[each.key].id
}

#############################
# IAM Roles and Policies    #
#############################

data "aws_iam_policy_document" "cluster_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "cluster" {
  name               = "${local.eks_name}-cluster-role"
  assume_role_policy = data.aws_iam_policy_document.cluster_assume_role.json

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSVpcResourceController" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
}

data "aws_iam_policy_document" "node_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "nodes" {
  name               = "${local.eks_name}-node-role"
  assume_role_policy = data.aws_iam_policy_document.node_assume_role.json

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "nodes_AmazonEKSWorkerNodePolicy" {
  role       = aws_iam_role.nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "nodes_AmazonEC2ContainerRegistryReadOnly" {
  role       = aws_iam_role.nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "nodes_AmazonEKS_CNI_Policy" {
  role       = aws_iam_role.nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "nodes_AmazonSSMManagedInstanceCore" {
  role       = aws_iam_role.nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

#####################
# EKS Control Plane #
#####################

resource "aws_eks_cluster" "this" {
  name     = local.eks_name
  role_arn = aws_iam_role.cluster.arn

  version = var.kubernetes_version

  vpc_config {
    subnet_ids = concat(
      [for subnet in aws_subnet.public : subnet.id],
      [for subnet in aws_subnet.private : subnet.id]
    )

    endpoint_public_access  = var.api_public_access
    endpoint_private_access = true
    public_access_cidrs     = var.public_access_cidrs
  }

  enabled_cluster_log_types = var.cluster_log_types

  kubernetes_network_config {
    service_ipv4_cidr = var.service_ipv4_cidr
  }

  tags = var.tags

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
    aws_iam_role_policy_attachment.cluster_AmazonEKSVpcResourceController
  ]
}

#####################
# Managed Node Groups #
#####################

resource "aws_eks_node_group" "this" {
  for_each = var.node_groups

  cluster_name    = aws_eks_cluster.this.name
  node_group_name = each.key
  node_role_arn   = aws_iam_role.nodes.arn

  subnet_ids = [
    for subnet in aws_subnet.private : subnet.id
  ]

  scaling_config {
    desired_size = each.value.desired_size
    max_size     = each.value.max_size
    min_size     = each.value.min_size
  }

  capacity_type = lookup(each.value, "capacity_type", "ON_DEMAND")

  instance_types = each.value.instance_types

  ami_type = lookup(each.value, "ami_type", null)

  update_config {
    max_unavailable = lookup(each.value, "max_unavailable", 1)
  }

  labels = lookup(each.value, "labels", {})

  taints = lookup(each.value, "taints", [])

  tags = merge(var.tags, {
    "kubernetes.io/cluster/${local.eks_name}" = "owned"
  })

  depends_on = [
    aws_iam_role_policy_attachment.nodes_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.nodes_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.nodes_AmazonEC2ContainerRegistryReadOnly
  ]
}

########################
# Outputs              #
########################

output "cluster_name" {
  value = aws_eks_cluster.this.name
}

output "cluster_endpoint" {
  value = aws_eks_cluster.this.endpoint
}

output "cluster_ca_certificate" {
  value = aws_eks_cluster.this.certificate_authority[0].data
}

output "vpc_id" {
  value = aws_vpc.eks.id
}

output "private_subnet_ids" {
  value = [for subnet in aws_subnet.private : subnet.id]
}

output "public_subnet_ids" {
  value = [for subnet in aws_subnet.public : subnet.id]
}


