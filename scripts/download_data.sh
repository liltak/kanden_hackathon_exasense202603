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

# COLMAP South Building dataset (building exterior, 128 images)
download_south_building() {
    local url="https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
    local dest="$DATA_DIR/colmap"
    mkdir -p "$dest"

    if [ -d "$dest/south-building" ]; then
        echo "[skip] COLMAP South Building already exists"
        return
    fi

    echo "[download] COLMAP South Building (~234MB, 128 images)"
    curl -L -o "$dest/south-building.zip" "$url"
    unzip -q "$dest/south-building.zip" -d "$dest/"
    rm "$dest/south-building.zip"
    echo "[done] COLMAP South Building → $dest/south-building"
}

# Usage
echo ""
echo "Available datasets:"
echo "  1) Mip-NeRF 360 garden (outdoor, ~200MB)"
echo "  2) Mip-NeRF 360 bicycle (outdoor, ~200MB)"
echo "  3) COLMAP south-building (building exterior, ~234MB)"
echo ""

case "${1:-garden}" in
    garden)          download_mipnerf360 "garden" ;;
    bicycle)         download_mipnerf360 "bicycle" ;;
    south-building)  download_south_building ;;
    all)
        download_mipnerf360 "garden"
        download_mipnerf360 "bicycle"
        download_south_building
        ;;
    *)
        echo "Usage: $0 [garden|bicycle|south-building|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Download complete ==="
