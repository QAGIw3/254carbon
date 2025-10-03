# Multi-Region Active-Active Deployment
# Terraform configuration for global infrastructure

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

  backend "s3" {
    bucket = "254carbon-terraform-state"
    key    = "multi-region/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  alias  = "us_east"
  region = "us-east-1"

  assume_role {
    role_arn = var.provider_role_arn
  }
}

provider "aws" {
  alias  = "eu_west"
  region = "eu-west-1"

  assume_role {
    role_arn = var.provider_role_arn
  }
}

provider "aws" {
  alias  = "apac"
  region = "ap-southeast-1"

  assume_role {
    role_arn = var.provider_role_arn
  }
}

# Define regions
locals {
  regions = {
    us_east = {
      aws_region = "us-east-1"
      name       = "US-East"
      cidr       = "10.0.0.0/16"
    }
    eu_west = {
      aws_region = "eu-west-1"
      name       = "EU-West"
      cidr       = "10.1.0.0/16"
    }
    apac = {
      aws_region = "ap-southeast-1"
      name       = "APAC"
      cidr       = "10.2.0.0/16"
    }
  }
}

# EKS Clusters in each region
module "eks_us_east" {
  source = "./modules/eks"

  providers = {
    aws = aws.us_east
  }

  region            = local.regions.us_east.aws_region
  cluster_name      = "254carbon-us-east"
  vpc_cidr          = local.regions.us_east.cidr
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

  node_groups = {
    general = {
      desired_size   = 3
      min_size       = 2
      max_size       = 10
      instance_types = ["t3.xlarge"]
    }
    ml = {
      desired_size   = 2
      min_size       = 1
      max_size       = 5
      instance_types = ["g4dn.xlarge"]
      ami_type       = "AL2_x86_64_GPU"
    }
  }

  tags = {
    Environment = "production"
    Region      = local.regions.us_east.name
    Project     = "254carbon"
  }
}

module "eks_eu_west" {
  source = "./modules/eks"

  providers = {
    aws = aws.eu_west
  }

  region            = local.regions.eu_west.aws_region
  cluster_name      = "254carbon-eu-west"
  vpc_cidr          = local.regions.eu_west.cidr
  availability_zones = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]

  node_groups = {
    general = {
      desired_size   = 3
      min_size       = 2
      max_size       = 10
      instance_types = ["t3.xlarge"]
    }
  }

  tags = {
    Environment = "production"
    Region      = local.regions.eu_west.name
    Project     = "254carbon"
  }
}

module "eks_apac" {
  source = "./modules/eks"

  providers = {
    aws = aws.apac
  }

  region            = local.regions.apac.aws_region
  cluster_name      = "254carbon-apac"
  vpc_cidr          = local.regions.apac.cidr
  availability_zones = ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"]

  node_groups = {
    general = {
      desired_size   = 2
      min_size       = 1
      max_size       = 8
      instance_types = ["t3.large"]
    }
  }

  tags = {
    Environment = "production"
    Region      = local.regions.apac.name
    Project     = "254carbon"
  }
}

# Global CloudFront distribution
resource "aws_cloudfront_distribution" "global_api" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "254Carbon Global API Distribution"
  price_class         = "PriceClass_All"
  
  origin {
    domain_name = module.eks_us_east.cluster_endpoint
    origin_id   = "us-east-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  origin {
    domain_name = module.eks_eu_west.cluster_endpoint
    origin_id   = "eu-west-origin"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  origin_group {
    origin_id = "api-origin-group"
    
    failover_criteria {
      status_codes = [500, 502, 503, 504]
    }
    
    member {
      origin_id = "us-east-origin"
    }
    
    member {
      origin_id = "eu-west-origin"
    }
  }
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "api-origin-group"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "X-API-Key"]
      
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 300
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = false
    acm_certificate_arn           = var.acm_certificate_arn
    ssl_support_method            = "sni-only"
    minimum_protocol_version      = "TLSv1.2_2021"
  }
}

# Route 53 for global traffic management
resource "aws_route53_zone" "main" {
  name = var.root_domain
}

resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "${var.api_subdomain}.${var.root_domain}"
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.global_api.domain_name
    zone_id                = aws_cloudfront_distribution.global_api.hosted_zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "regional" {
  for_each = var.regional_api_endpoints

  zone_id        = aws_route53_zone.main.zone_id
  name           = "${each.key}.${var.api_subdomain}.${var.root_domain}"
  type           = "A"
  set_identifier = each.key

  latency_routing_policy {
    region = each.key
  }

  alias {
    name                   = each.value.domain_name
    zone_id                = each.value.hosted_zone_id
    evaluate_target_health = true
  }
}

# Global Aurora for cross-region replication
module "aurora_global" {
  source = "./modules/aurora-global"

  global_cluster_identifier = "254carbon-global"
  database_name             = "market_intelligence"
  primary_region            = local.regions.us_east.aws_region
  secondary_regions         = [local.regions.eu_west.aws_region, local.regions.apac.aws_region]
  master_username           = var.db_master_username
  master_password           = var.db_master_password
  kms_key_id                = var.db_kms_key_arn

  primary_vpc_cidr           = local.regions.us_east.cidr
  primary_availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  primary_private_subnets    = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  primary_public_subnets     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  primary_allowed_cidrs      = [local.regions.us_east.cidr]

  secondary_vpc_cidrs = {
    (local.regions.eu_west.aws_region) = local.regions.eu_west.cidr
    (local.regions.apac.aws_region)    = local.regions.apac.cidr
  }

  secondary_availability_zones = {
    (local.regions.eu_west.aws_region) = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
    (local.regions.apac.aws_region)    = ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"]
  }

  secondary_private_subnets = {
    (local.regions.eu_west.aws_region) = ["10.1.1.0/24", "10.1.2.0/24", "10.1.3.0/24"]
    (local.regions.apac.aws_region)    = ["10.2.1.0/24", "10.2.2.0/24", "10.2.3.0/24"]
  }

  secondary_public_subnets = {
    (local.regions.eu_west.aws_region) = ["10.1.101.0/24", "10.1.102.0/24", "10.1.103.0/24"]
    (local.regions.apac.aws_region)    = ["10.2.101.0/24", "10.2.102.0/24", "10.2.103.0/24"]
  }

  secondary_allowed_cidrs = {
    (local.regions.eu_west.aws_region) = [local.regions.eu_west.cidr]
    (local.regions.apac.aws_region)    = [local.regions.apac.cidr]
  }

  tags = {
    Environment = "production"
    Project     = "254carbon"
  }
}

# ElastiCache Redis Global Datastore
resource "aws_elasticache_global_replication_group" "redis" {
  global_replication_group_id_suffix   = "254carbon"
  primary_replication_group_id         = aws_elasticache_replication_group.primary.id
  global_replication_group_description = "Global Redis for 254Carbon"
}

# Outputs
output "cloudfront_domain" {
  value = aws_cloudfront_distribution.global_api.domain_name
}

output "regional_clusters" {
  value = {
    us_east = {
      endpoint = module.eks_us_east.cluster_endpoint
      vpc_id   = module.eks_us_east.vpc_id
    }
    eu_west = {
      endpoint = module.eks_eu_west.cluster_endpoint
      vpc_id   = module.eks_eu_west.vpc_id
    }
    apac = {
      endpoint = module.eks_apac.cluster_endpoint
      vpc_id   = module.eks_apac.vpc_id
    }
  }
}

