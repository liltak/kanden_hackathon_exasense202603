variable "aws_profile" {
  description = "AWS SSO profile name"
  type        = string
  default     = "default"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "instance_type" {
  description = "EC2 instance type (no GPU needed for Next.js)"
  type        = string
  default     = "t3.small"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 20
}

variable "project_name" {
  description = "Project name used for resource tagging"
  type        = string
  default     = "exasense-portal"
}

variable "subdomain" {
  description = "Subdomain name for the portal"
  type        = string
  default     = "exasense-portal"
}

variable "alb_name" {
  description = "Name of the existing ALB"
  type        = string
  default     = "your-alb-name"
}

variable "vpc_name" {
  description = "Name tag of the existing VPC"
  type        = string
  default     = "your-vpc-name"
}

variable "alb_rule_priority" {
  description = "Priority for the ALB listener rule (must be unique)"
  type        = number
  default     = 311
}

variable "iam_instance_profile" {
  description = "Existing IAM instance profile name for Session Manager access"
  type        = string
  default     = "your-instance-profile"
}

variable "github_token" {
  description = "GitHub Personal Access Token (passed via env or tfvars)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository (owner/repo)"
  type        = string
  default     = "your-org/your-repo"
}
