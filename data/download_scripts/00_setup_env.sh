#!/usr/bin/env bash
# =============================================================================
# Step 0: Install prerequisites for dataset download
#
# Only needs seqtk (standalone binary) — no Python venv required.
# All Python scripts use stdlib only.
#
# Usage:
#   bash data/download_scripts/00_setup_env.sh
# =============================================================================
set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-/scratch/10906/arisk/evo_locking_data}"
LOCAL_BIN="$SCRATCH_DIR/tools/bin"
mkdir -p "$LOCAL_BIN"

if command -v seqtk &>/dev/null; then
    echo "seqtk already installed: $(which seqtk)"
elif [ -x "$LOCAL_BIN/seqtk" ]; then
    echo "seqtk found at $LOCAL_BIN/seqtk"
    echo "Add to PATH: export PATH=\"$LOCAL_BIN:\$PATH\""
else
    echo "Building seqtk from source..."
    BUILD_DIR="$SCRATCH_DIR/tools/seqtk_build"
    rm -rf "$BUILD_DIR"
    git clone https://github.com/lh3/seqtk.git "$BUILD_DIR"
    cd "$BUILD_DIR"
    make
    cp seqtk "$LOCAL_BIN/"
    cd -
    rm -rf "$BUILD_DIR"
    echo "Installed seqtk to $LOCAL_BIN/seqtk"
fi

export PATH="$LOCAL_BIN:$PATH"

echo ""
echo "Prerequisites OK."
echo "  seqtk:   $(which seqtk)"
echo "  python3: $(which python3)"
echo "  wget:    $(which wget)"
echo "  curl:    $(which curl)"
echo ""
echo "If running scripts individually, first run:"
echo "  export PATH=\"$LOCAL_BIN:\$PATH\""
