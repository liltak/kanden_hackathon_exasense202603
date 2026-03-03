terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  profile = var.aws_profile
  region  = var.aws_region
}

# ============================================================
# Existing aie-demo VPC
# ============================================================
data "aws_vpc" "aie_demo" {
  filter {
    name   = "tag:Name"
    values = [var.vpc_name]
  }
}

data "aws_subnets" "aie_demo_private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.aie_demo.id]
  }
  filter {
    name   = "tag:Name"
    values = ["aie-demo-private"]
  }
  filter {
    name   = "availability-zone"
    values = ["ap-northeast-1d"]
  }
}

# ============================================================
# Existing ALB (aie-demo) and HTTPS:443 Listener
# ============================================================
data "aws_lb" "aie_demo" {
  name = var.alb_name
}

data "aws_lb_listener" "https" {
  load_balancer_arn = data.aws_lb.aie_demo.arn
  port              = 443
}

# ============================================================
# SSH Key Pair (for SSM port-forwarding + SSH)
# ============================================================
resource "tls_private_key" "ssh" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "portal" {
  key_name   = "${var.project_name}-key"
  public_key = tls_private_key.ssh.public_key_openssh
}

resource "local_file" "private_key" {
  content         = tls_private_key.ssh.private_key_pem
  filename        = "${abspath(path.module)}/${var.project_name}.pem"
  file_permission = "0400"
}

# ============================================================
# Security Group (ALB only)
# ============================================================
resource "aws_security_group" "portal" {
  name        = "${var.project_name}-sg"
  description = "Security group for ExaSense Portal"
  vpc_id      = data.aws_vpc.aie_demo.id

  ingress {
    description     = "Allow from ALB"
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    security_groups = data.aws_lb.aie_demo.security_groups
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

# ============================================================
# AMI: Ubuntu 22.04 (no GPU needed)
# ============================================================
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# ============================================================
# EC2 Instance (private subnet, SSM access)
# ============================================================
resource "aws_instance" "portal" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.portal.key_name
  vpc_security_group_ids = [aws_security_group.portal.id]
  subnet_id              = data.aws_subnets.aie_demo_private.ids[0]
  iam_instance_profile   = var.iam_instance_profile

  associate_public_ip_address = false

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = file("${path.module}/user_data.sh")

  tags = {
    Name = var.project_name
  }
}

# ============================================================
# ALB Target Group
# ============================================================
resource "aws_lb_target_group" "portal" {
  name     = "${var.project_name}-tg"
  port     = 3000
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.aie_demo.id

  health_check {
    path                = "/api/health"
    protocol            = "HTTP"
    port                = "3000"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  tags = {
    Name = "${var.project_name}-tg"
  }
}

resource "aws_lb_target_group_attachment" "portal" {
  target_group_arn = aws_lb_target_group.portal.arn
  target_id        = aws_instance.portal.id
  port             = 3000
}

# ============================================================
# ALB Listener Rule (host header routing)
# ============================================================
# ============================================================
# IAM: Bedrock InvokeModel policy for AI summaries
# ============================================================
data "aws_iam_instance_profile" "portal" {
  name = var.iam_instance_profile
}

resource "aws_iam_role_policy" "bedrock" {
  name = "${var.project_name}-bedrock"
  role = data.aws_iam_instance_profile.portal.role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/anthropic.*",
          "arn:aws:bedrock:*:*:inference-profile/us.anthropic.*",
          "arn:aws:bedrock:*:*:inference-profile/eu.anthropic.*",
          "arn:aws:bedrock:*:*:inference-profile/ap.anthropic.*"
        ]
      }
    ]
  })
}

resource "aws_lb_listener_rule" "portal" {
  listener_arn = data.aws_lb_listener.https.arn
  priority     = var.alb_rule_priority

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.portal.arn
  }

  condition {
    host_header {
      values = ["${var.subdomain}.demo.exwzd.ai"]
    }
  }

  tags = {
    Name = var.project_name
  }
}
