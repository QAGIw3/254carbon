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
  
  region      = local.regions.us_east.aws_region
  cluster_name = "254carbon-us-east"
  vpc_cidr    = local.regions.us_east.cidr
  
  node_groups = {
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10
      instance_types = ["t3.xlarge"]
    }
    ml = {
      desired_size = 2
      min_size     = 1
      max_size     = 5
      instance_types = ["g4dn.xlarge"]  # GPU instances
    }
  }
  
  providers = {
    aws = aws.us_east
  }
}

module "eks_eu_west" {
  source = "./modules/eks"
  
  region      = local.regions.eu_west.aws_region
  cluster_name = "254carbon-eu-west"
  vpc_cidr    = local.regions.eu_west.cidr
  
  node_groups = {
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10
      instance_types = ["t3.xlarge"]
    }
  }
  
  providers = {
    aws = aws.eu_west
  }
}

module "eks_apac" {
  source = "./modules/eks"
  
  region      = local.regions.apac.aws_region
  cluster_name = "254carbon-apac"
  vpc_cidr    = local.regions.apac.cidr
  
  node_groups = {
    general = {
      desired_size = 2
      min_size     = 1
      max_size     = 8
      instance_types = ["t3.large"]
    }
  }
  
  providers = {
    aws = aws.apac
  }
}

# Global CloudFront distribution
resource "aws_cloudfront_distribution" "global_api" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "254Carbon Global API Distribution"
  price_class         = "PriceClass_All"
  
  origin {
    domain_name = module.eks_us_east.api_endpoint
    origin_id   = "us-east-origin"
    
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
    acm_certificate_arn           = aws_acm_certificate.global.arn
    ssl_support_method            = "sni-only"
    minimum_protocol_version      = "TLSv1.2_2021"
  }
}

# Route 53 for global traffic management
resource "aws_route53_zone" "main" {
  name = "254carbon.ai"
}

resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.254carbon.ai"
  type    = "A"
  
  alias {
    name                   = aws_cloudfront_distribution.global_api.domain_name
    zone_id                = aws_cloudfront_distribution.global_api.hosted_zone_id
    evaluate_target_health = true
  }
}

# Latency-based routing for regions
resource "aws_route53_record" "us_east" {
  zone_id        = aws_route53_zone.main.zone_id
  name           = "us.api.254carbon.ai"
  type           = "A"
  set_identifier = "us-east-1"
  
  latency_routing_policy {
    region = "us-east-1"
  }
  
  alias {
    name                   = module.eks_us_east.api_endpoint
    zone_id                = module.eks_us_east.zone_id
    evaluate_target_health = true
  }
}

# Global Aurora for cross-region replication
module "aurora_global" {
  source = "./modules/aurora-global"
  
  global_cluster_identifier = "254carbon-global"
  engine                    = "aurora-postgresql"
  engine_version            = "15.3"
  database_name             = "market_intelligence"
  
  primary_region    = "us-east-1"
  secondary_regions = ["eu-west-1", "ap-southeast-1"]
  
  instance_class = "db.r6g.xlarge"
}

# ElastiCache Redis Global Datastore
resource "aws_elasticache_global_replication_group" "redis" {
  global_replication_group_id_suffix = "254carbon"
  primary_replication_group_id       = aws_elasticache_replication_group.us_east.id
  
  global_replication_group_description = "Global Redis for 254Carbon"
}

# Outputs
output "cloudfront_domain" {
  value = aws_cloudfront_distribution.global_api.domain_name
}

output "api_endpoint" {
  value = "https://api.254carbon.ai"
}

output "regional_endpoints" {
  value = {
    us_east = module.eks_us_east.api_endpoint
    eu_west = module.eks_eu_west.api_endpoint
    apac    = module.eks_apac.api_endpoint
  }
}

