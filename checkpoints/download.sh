#!/usr/bin/env bash
# Download all model checkpoints needed for HVUE LoRA fine-tuning.
#
# Source: H100 node (this machine). Adjust HOST if your SSH config differs.
# Usage on the target cluster:
#   cd glm-locking && bash checkpoints/download.sh
set -euo pipefail

HOST="${1:-brev-d3p1gsp1f}"   # source machine hostname (SSH-able)
SRC_DIR="/home/nvidia/glm-locking/checkpoints"
DST_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Pulling checkpoints from ${HOST}:${SRC_DIR} → ${DST_DIR}"
echo "Total: ~98 GB (may take 10–30 min depending on network)"
echo ""

rsync -avP --progress "${HOST}:${SRC_DIR}/" "${DST_DIR}/"

echo ""
echo "Done. Checkpoints in ${DST_DIR}/"
ls -lh "${DST_DIR}/"
