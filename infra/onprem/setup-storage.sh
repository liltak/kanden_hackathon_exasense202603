#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ExaSense On-Prem Storage Setup
#
# Sets up MinIO for S3-compatible object storage and NFS for model weights.
# Run on the storage/GPU server after k3s setup.
#
# Usage: sudo ./setup-storage.sh
# =============================================================================

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
err() { log "ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || err "Run as root: sudo $0"

log "=== ExaSense Storage Setup ==="

# --- Data directories ---
MINIO_DATA="/data/exasense/minio"
MODEL_CACHE="/data/exasense/model-cache"
NFS_EXPORT="/data/exasense/nfs"

for dir in "$MINIO_DATA" "$MODEL_CACHE" "$NFS_EXPORT"; do
    mkdir -p "$dir"
    log "Created $dir"
done

# --- MinIO (standalone binary for on-prem) ---
log "Setting up MinIO..."

if ! command -v minio &>/dev/null; then
    wget -q https://dl.min.io/server/minio/release/linux-amd64/minio -O /usr/local/bin/minio
    chmod +x /usr/local/bin/minio
    log "MinIO binary installed"
fi

# MinIO systemd service
cat > /etc/systemd/system/minio.service << 'EOF'
[Unit]
Description=MinIO Object Storage
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root
EnvironmentFile=/etc/default/minio
ExecStart=/usr/local/bin/minio server /data/exasense/minio --console-address ":9001"
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

# MinIO environment
MINIO_ACCESS=${MINIO_ACCESS_KEY:-"exasense"}
MINIO_SECRET=${MINIO_SECRET_KEY:-"$(openssl rand -base64 24)"}

cat > /etc/default/minio << EOF
MINIO_ROOT_USER=${MINIO_ACCESS}
MINIO_ROOT_PASSWORD=${MINIO_SECRET}
MINIO_VOLUMES="/data/exasense/minio"
EOF

systemctl daemon-reload
systemctl enable --now minio
log "MinIO started on :9000 (console :9001)"
log "  Access Key: $MINIO_ACCESS"
log "  Secret Key: $MINIO_SECRET"

# Create default buckets
sleep 3
if command -v mc &>/dev/null || {
    wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc
    chmod +x /usr/local/bin/mc
}; then
    mc alias set local http://localhost:9000 "$MINIO_ACCESS" "$MINIO_SECRET" 2>/dev/null || true
    mc mb local/exasense-meshes --ignore-existing 2>/dev/null || true
    mc mb local/exasense-reports --ignore-existing 2>/dev/null || true
    mc mb local/exasense-models --ignore-existing 2>/dev/null || true
    log "Default buckets created"
fi

# --- NFS for model weights (shared across workers) ---
log "Setting up NFS..."

apt-get update -qq && apt-get install -y -qq nfs-kernel-server

echo "$NFS_EXPORT *(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports
exportfs -ra
systemctl enable --now nfs-kernel-server
log "NFS export: $NFS_EXPORT"

# --- Download model weights ---
log "Downloading model weights to $MODEL_CACHE..."
log "  (This may take a while for large models)"

# Create HuggingFace cache directory
mkdir -p "$MODEL_CACHE/huggingface"

log ""
log "=== Storage Setup Complete ==="
log ""
log "MinIO:  http://$(hostname -I | awk '{print $1}'):9000"
log "Console: http://$(hostname -I | awk '{print $1}'):9001"
log "NFS:    $NFS_EXPORT"
log ""
log "Next steps:"
log "  1. Configure bucket replication to cloud MinIO"
log "  2. Pre-download model weights: HF_HOME=$MODEL_CACHE/huggingface huggingface-cli download <model>"
log "  3. Mount NFS on worker nodes: mount -t nfs $(hostname -I | awk '{print $1}'):$NFS_EXPORT /data/exasense/model-cache"
