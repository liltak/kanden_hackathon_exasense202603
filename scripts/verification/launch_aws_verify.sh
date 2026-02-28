#!/usr/bin/env bash
# Launch Phase 1-2 verification on AWS GPU spot instance.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - EC2 key pair created
#   - VPC/Subnet IDs known
#
# Usage:
#   ./launch_aws_verify.sh                      # Interactive prompts
#   ./launch_aws_verify.sh --quick              # Use defaults, just provide key pair
#   ./launch_aws_verify.sh --teardown           # Delete the stack
#
# Cost estimate: g5.xlarge spot ≈ $0.30-0.50/hr, ~$1 for full verification
set -euo pipefail

STACK_NAME="exasense-gpu-verify"
TEMPLATE="$(cd "$(dirname "$0")/../../infra" && pwd)/gpu-verify-stack.yaml"
REGION="${AWS_DEFAULT_REGION:-ap-northeast-1}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Teardown ──────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--teardown" ]; then
    info "Deleting stack $STACK_NAME..."
    aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
    aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION"
    ok "Stack deleted."
    exit 0
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
info "Checking prerequisites..."

if ! command -v aws &>/dev/null; then
    err "AWS CLI not found. Install: https://aws.amazon.com/cli/"
    exit 1
fi

if ! aws sts get-caller-identity &>/dev/null; then
    err "AWS credentials not configured. Run: aws configure"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
info "AWS Account: $ACCOUNT_ID, Region: $REGION"

# ── Find AMI ──────────────────────────────────────────────────────────────────
info "Finding Deep Learning AMI..."
DL_AMI=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text 2>/dev/null || echo "")

if [ -z "$DL_AMI" ] || [ "$DL_AMI" = "None" ]; then
    info "DL AMI not found, trying generic Ubuntu GPU AMI..."
    DL_AMI=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
            "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text 2>/dev/null || echo "")
fi

if [ -z "$DL_AMI" ] || [ "$DL_AMI" = "None" ]; then
    err "No suitable Deep Learning AMI found in $REGION"
    err "Try a different region (us-east-1, us-west-2) or specify AMI manually"
    exit 1
fi
ok "AMI: $DL_AMI"

# ── Get parameters ────────────────────────────────────────────────────────────
# Key pair
KEY_PAIRS=$(aws ec2 describe-key-pairs --region "$REGION" --query 'KeyPairs[].KeyName' --output text)
if [ -z "$KEY_PAIRS" ]; then
    err "No EC2 key pairs found. Create one first:"
    err "  aws ec2 create-key-pair --key-name exasense --query 'KeyMaterial' --output text > exasense.pem"
    exit 1
fi
echo ""
echo "Available key pairs: $KEY_PAIRS"
read -rp "Key pair name [$(echo "$KEY_PAIRS" | awk '{print $1}')]: " KEY_PAIR
KEY_PAIR="${KEY_PAIR:-$(echo "$KEY_PAIRS" | awk '{print $1}')}"

# VPC
DEFAULT_VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "")
if [ -z "$DEFAULT_VPC" ] || [ "$DEFAULT_VPC" = "None" ]; then
    read -rp "VPC ID: " VPC_ID
else
    read -rp "VPC ID [$DEFAULT_VPC]: " VPC_ID
    VPC_ID="${VPC_ID:-$DEFAULT_VPC}"
fi

# Subnet
DEFAULT_SUBNET=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
    --query 'Subnets[0].SubnetId' --output text 2>/dev/null || echo "")
if [ -z "$DEFAULT_SUBNET" ] || [ "$DEFAULT_SUBNET" = "None" ]; then
    DEFAULT_SUBNET=$(aws ec2 describe-subnets --region "$REGION" \
        --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[0].SubnetId' --output text)
fi
read -rp "Subnet ID [$DEFAULT_SUBNET]: " SUBNET_ID
SUBNET_ID="${SUBNET_ID:-$DEFAULT_SUBNET}"

# ── Check spot pricing ───────────────────────────────────────────────────────
info "Checking g5.xlarge spot price..."
SPOT_PRICE=$(aws ec2 describe-spot-price-history \
    --region "$REGION" \
    --instance-types g5.xlarge \
    --product-descriptions "Linux/UNIX" \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
    --query 'SpotPriceHistory[0].SpotPrice' \
    --output text 2>/dev/null || echo "N/A")

if [ "$SPOT_PRICE" = "N/A" ] || [ -z "$SPOT_PRICE" ]; then
    info "g5.xlarge not available in $REGION, trying g4dn.xlarge..."
    INSTANCE_TYPE="g4dn.xlarge"
    SPOT_PRICE=$(aws ec2 describe-spot-price-history \
        --region "$REGION" \
        --instance-types g4dn.xlarge \
        --product-descriptions "Linux/UNIX" \
        --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
        --query 'SpotPriceHistory[0].SpotPrice' \
        --output text 2>/dev/null || echo "N/A")
else
    INSTANCE_TYPE="g5.xlarge"
fi

info "Instance type: $INSTANCE_TYPE, Current spot price: \$$SPOT_PRICE/hr"
echo ""
read -rp "Proceed with deployment? [y/N]: " CONFIRM
if [ "${CONFIRM,,}" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# ── Deploy stack ──────────────────────────────────────────────────────────────
info "Deploying CloudFormation stack: $STACK_NAME..."

aws cloudformation deploy \
    --region "$REGION" \
    --template-file "$TEMPLATE" \
    --stack-name "$STACK_NAME" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
        KeyPairName="$KEY_PAIR" \
        VpcId="$VPC_ID" \
        SubnetId="$SUBNET_ID" \
        MaxSpotPrice="0.80" \
        AutoTerminateMinutes=120 \
    --tags Key=Project,Value=exasense

# Get outputs
info "Getting instance details..."
OUTPUTS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].Outputs' --output json)

INSTANCE_ID=$(echo "$OUTPUTS" | python3 -c "import json,sys; d={o['OutputKey']:o['OutputValue'] for o in json.load(sys.stdin)}; print(d.get('InstanceId',''))")
PUBLIC_IP=$(echo "$OUTPUTS" | python3 -c "import json,sys; d={o['OutputKey']:o['OutputValue'] for o in json.load(sys.stdin)}; print(d.get('PublicIP',''))")

echo ""
ok "=== Deployment Complete ==="
echo ""
echo "Instance: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Commands:"
echo "  SSH:    ssh -i ${KEY_PAIR}.pem ubuntu@${PUBLIC_IP}"
echo "  Logs:   ssh -i ${KEY_PAIR}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/exasense-setup.log'"
echo "  Status: aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name'"
echo ""
echo "  Teardown: $0 --teardown"
echo ""
info "Instance will auto-terminate in 120 minutes."
info "Setup log: /var/log/exasense-setup.log"
