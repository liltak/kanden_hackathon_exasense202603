#!/bin/bash
set -euxo pipefail

LOG_FILE="/var/log/user_data.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Starting ExaSense Portal setup: $(date) ==="

# ============================================================
# System update & Node.js 20 LTS install
# ============================================================
apt-get update -y
apt-get install -y curl ca-certificates gnupg

# Node.js 20 via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

echo "Node.js version: $(node -v)"
echo "npm version: $(npm -v)"

# ============================================================
# App directory setup
# ============================================================
APP_DIR="/home/ubuntu/exasense-portal"
mkdir -p "$APP_DIR"
chown -R ubuntu:ubuntu "$APP_DIR"

# ============================================================
# systemd service for Next.js app
# ============================================================
cat > /etc/systemd/system/exasense-portal.service << 'UNIT'
[Unit]
Description=ExaSense Portal Next.js App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/exasense-portal
EnvironmentFile=/home/ubuntu/exasense-portal/.env
ExecStart=/usr/bin/node server.js
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable exasense-portal.service
# Don't start yet — app code will be deployed via deploy.sh

echo "=== Setup complete: $(date) ==="
