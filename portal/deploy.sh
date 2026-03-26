#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$SCRIPT_DIR/terraform"
LOCAL_PORT=2222
AWS_PROFILE="${AWS_PROFILE:-default}"
AWS_REGION="ap-northeast-1"

# Get outputs from Terraform
echo "Reading Terraform outputs..."
INSTANCE_ID=$(terraform -chdir="$TF_DIR" output -raw instance_id)
KEY_PATH=$(terraform -chdir="$TF_DIR" output -raw private_key_path)
APP_URL=$(terraform -chdir="$TF_DIR" output -raw app_url)

REMOTE_DIR="/home/ubuntu/exasense-portal"
SSH_OPTS="-i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $LOCAL_PORT"

# Start SSM port forwarding (SSH tunnel)
echo "Starting SSM port forwarding to $INSTANCE_ID ..."
aws ssm start-session \
  --target "$INSTANCE_ID" \
  --document-name AWS-StartPortForwardingSession \
  --parameters "{\"portNumber\":[\"22\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}" \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" &
SSM_PID=$!

# Wait for tunnel to establish
echo "Waiting for SSM tunnel..."
for i in $(seq 1 30); do
  if nc -z localhost "$LOCAL_PORT" 2>/dev/null; then
    echo "Tunnel established."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: SSM tunnel failed to establish after 30s"
    kill "$SSM_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

cleanup() {
  echo "Closing SSM tunnel..."
  kill "$SSM_PID" 2>/dev/null || true
  wait "$SSM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Build locally first
echo "Building Next.js app..."
cd "$SCRIPT_DIR"
npm run build

# Create remote directories
echo "Creating remote directories..."
ssh $SSH_OPTS "ubuntu@localhost" "mkdir -p $REMOTE_DIR/.next/static"

# Sync standalone build output
echo "Deploying standalone build..."
rsync -avz --delete -e "ssh $SSH_OPTS" \
  "$SCRIPT_DIR/.next/standalone/" \
  "ubuntu@localhost:$REMOTE_DIR/"

rsync -avz --delete -e "ssh $SSH_OPTS" \
  "$SCRIPT_DIR/.next/static/" \
  "ubuntu@localhost:$REMOTE_DIR/.next/static/"

# Sync public directory if it exists
if [ -d "$SCRIPT_DIR/public" ]; then
  rsync -avz --delete -e "ssh $SSH_OPTS" \
    "$SCRIPT_DIR/public/" \
    "ubuntu@localhost:$REMOTE_DIR/public/"
fi

# Create .env file on remote
echo "Setting up environment variables..."
ssh $SSH_OPTS "ubuntu@localhost" << EOF
cat > $REMOTE_DIR/.env << 'ENVFILE'
NODE_ENV=production
PORT=3000
HOSTNAME=0.0.0.0
GITHUB_TOKEN=${GITHUB_TOKEN:-}
GITHUB_REPO=${GITHUB_REPO:-your-org/your-repo}
AWS_REGION=$AWS_REGION
ENVFILE
EOF

# Restart the service
echo "Restarting application service..."
ssh $SSH_OPTS "ubuntu@localhost" << 'EOF'
sudo systemctl restart exasense-portal.service
sleep 2
sudo systemctl status exasense-portal.service --no-pager || true
EOF

echo ""
echo "Deploy complete!"
echo "App URL: $APP_URL"
echo "(Access via VPN if applicable)"
