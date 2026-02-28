#!/usr/bin/env bash
# Download test datasets for ExaSense
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DATA_DIR"

echo "=== ExaSense Data Downloader ==="

# Mip-NeRF 360 dataset (small-scale test for Phase 1)
download_mipnerf360() {
    local scene="${1:-garden}"
    local url="http://storage.googleapis.com/gresearch/refraw360/${scene}.zip"
    local dest="$DATA_DIR/mipnerf360"
    mkdir -p "$dest"

    if [ -d "$dest/$scene" ]; then
        echo "[skip] Mip-NeRF 360 '$scene' already exists"
        return
    fi

    echo "[download] Mip-NeRF 360 '$scene' from $url"
    curl -L -o "$dest/${scene}.zip" "$url"
    unzip -q "$dest/${scene}.zip" -d "$dest/"
    rm "$dest/${scene}.zip"
    echo "[done] Mip-NeRF 360 '$scene' → $dest/$scene"
}

# Usage
echo ""
echo "Available datasets:"
echo "  1) Mip-NeRF 360 garden (small, ~200MB)"
echo "  2) Mip-NeRF 360 bicycle (small, ~200MB)"
echo ""

case "${1:-garden}" in
    garden)   download_mipnerf360 "garden" ;;
    bicycle)  download_mipnerf360 "bicycle" ;;
    all)
        download_mipnerf360 "garden"
        download_mipnerf360 "bicycle"
        ;;
    *)
        echo "Usage: $0 [garden|bicycle|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Download complete ==="
