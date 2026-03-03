output "app_url" {
  description = "URL to access the portal (via ALB + CorpVPN)"
  value       = "https://${var.subdomain}.demo.exwzd.ai"
}

output "instance_id" {
  description = "EC2 instance ID (for SSM access)"
  value       = aws_instance.portal.id
}

output "private_ip" {
  description = "Private IP address of the EC2 instance"
  value       = aws_instance.portal.private_ip
}

output "ssm_command" {
  description = "SSM command to connect to the instance"
  value       = "aws ssm start-session --target ${aws_instance.portal.id} --profile ${var.aws_profile} --region ${var.aws_region}"
}

output "private_key_path" {
  description = "Path to the SSH private key (for SSM port-forwarding + SSH)"
  value       = local_file.private_key.filename
}

output "target_group_arn" {
  description = "ARN of the target group"
  value       = aws_lb_target_group.portal.arn
}
