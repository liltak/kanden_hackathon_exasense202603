#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ExaSense On-Prem k3s Setup Script
#
# Sets up k3s + NVIDIA GPU Operator + Tailscale on an on-premises GPU server.
# Requirements: Ubuntu 22.04+, NVIDIA drivers installed, internet access.
#
# Usage:
#   sudo ./setup-k3s.sh [--join <server-url> <token>]
#
# Options:
#   --join <url> <token>  Join existing k3s cluster instead of creating new one
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/exasense-k3s-setup.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
err() { log "ERROR: $*" >&2; exit 1; }

# --- Pre-flight checks ---
[[ $EUID -eq 0 ]] || err "Run as root: sudo $0"

log "=== ExaSense On-Prem k3s Setup ==="

# Check GPU
if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Install NVIDIA drivers first."
fi
log "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG_FILE"

# --- Parse arguments ---
JOIN_MODE=false
JOIN_URL=""
JOIN_TOKEN=""
if [[ "${1:-}" == "--join" ]]; then
    JOIN_MODE=true
    JOIN_URL="${2:-}"
    JOIN_TOKEN="${3:-}"
    [[ -n "$JOIN_URL" && -n "$JOIN_TOKEN" ]] || err "Usage: $0 --join <server-url> <token>"
fi

# --- Install k3s ---
log "Installing k3s..."
if [[ "$JOIN_MODE" == true ]]; then
    curl -sfL https://get.k3s.io | K3S_URL="$JOIN_URL" K3S_TOKEN="$JOIN_TOKEN" sh -s - agent \
        --node-label="exasense/role=gpu-worker" \
        --node-label="nvidia.com/gpu.present=true"
    log "k3s agent joined cluster at $JOIN_URL"
else
    curl -sfL https://get.k3s.io | sh -s - server \
        --disable=traefik \
        --write-kubeconfig-mode=644 \
        --node-label="exasense/role=gpu-worker" \
        --node-label="nvidia.com/gpu.present=true"

    # Wait for k3s
    log "Waiting for k3s to be ready..."
    until kubectl get nodes &>/dev/null; do sleep 2; done
    log "k3s server ready"

    # Save join token
    JOIN_TOKEN=$(cat /var/lib/rancher/k3s/server/node-token)
    log "Join token for additional nodes: $JOIN_TOKEN"
    log "Join command: curl -sfL https://get.k3s.io | K3S_URL=https://$(hostname -I | awk '{print $1}'):6443 K3S_TOKEN=$JOIN_TOKEN sh -s - agent"
fi

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# --- Install Helm ---
if ! command -v helm &>/dev/null; then
    log "Installing Helm..."
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

# --- Install NVIDIA GPU Operator ---
if [[ "$JOIN_MODE" == false ]]; then
    log "Installing NVIDIA GPU Operator..."

    # Add Helm repos
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    helm repo update

    # Install GPU Operator (includes device plugin, DCGM, container toolkit)
    helm install --wait --generate-name \
        -n gpu-operator --create-namespace \
        nvidia/gpu-operator \
        --set driver.enabled=false \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=true \
        --set dcgmExporter.enabled=true

    log "GPU Operator installed. Waiting for device plugin..."
    kubectl wait --for=condition=ready pod -l app=nvidia-device-plugin-daemonset \
        -n gpu-operator --timeout=300s || log "WARNING: GPU device plugin not ready yet"
fi

# --- Install KEDA ---
if [[ "$JOIN_MODE" == false ]]; then
    log "Installing KEDA for queue-based autoscaling..."
    helm repo add kedacore https://kedacore.github.io/charts
    helm repo update
    helm install keda kedacore/keda -n keda --create-namespace --wait
    log "KEDA installed"
fi

# --- Install Tailscale ---
log "Installing Tailscale..."
if ! command -v tailscale &>/dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

TAILSCALE_AUTHKEY="${TAILSCALE_AUTHKEY:-}"
if [[ -n "$TAILSCALE_AUTHKEY" ]]; then
    tailscale up --authkey="$TAILSCALE_AUTHKEY" --advertise-tags=tag:onprem-worker --accept-routes
    log "Tailscale connected"
else
    log "Set TAILSCALE_AUTHKEY env var and run: tailscale up --authkey=\$TAILSCALE_AUTHKEY --advertise-tags=tag:onprem-worker --accept-routes"
fi

# --- Install Traefik (server only) ---
if [[ "$JOIN_MODE" == false ]]; then
    log "Installing Traefik ingress controller..."
    helm repo add traefik https://traefik.github.io/charts
    helm repo update
    helm install traefik traefik/traefik -n traefik --create-namespace \
        --set ports.websecure.tls.enabled=true
    log "Traefik installed"
fi

# --- Verify ---
log ""
log "=== Setup Complete ==="
log ""
kubectl get nodes -o wide 2>/dev/null || true
echo ""
log "GPU status:"
kubectl describe node | grep -A5 "nvidia.com/gpu" 2>/dev/null || log "GPU resources not yet available (GPU Operator may still be initializing)"
echo ""
log "Next steps:"
log "  1. Ensure Tailscale is connected: tailscale status"
log "  2. Deploy ExaSense: helm install exasense infra/helm/exasense/ -f infra/helm/exasense/values-production.yaml"
log "  3. Verify GPU workers: kubectl get pods -n exasense"
