#!/bin/bash
# ============================================================
# Label Studio — Audio Waveform Analysis for DSP501
# ============================================================
# Usage: bash tools/label-studio/start.sh
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$PROJECT_ROOT/venv/bin"
LS_DIR="$PROJECT_ROOT/tools/label-studio"

echo "============================================"
echo "  DSP501 — Label Studio Setup"
echo "============================================"

# Step 1: Prepare audio samples
echo ""
echo "[1/3] Preparing audio samples..."
"$VENV/python3" "$LS_DIR/prepare_tasks.py" --samples 5

# Step 2: Set environment
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$LS_DIR"
export LABEL_STUDIO_BASE_DATA_DIR="$LS_DIR/ls_data"
export LABEL_STUDIO_PORT=8080

echo ""
echo "[2/3] Configuration:"
echo "  Port: $LABEL_STUDIO_PORT"
echo "  Data root: $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
echo ""
echo "[3/3] Starting Label Studio..."
echo ""
echo "============================================"
echo "  INSTRUCTIONS (Hướng dẫn):"
echo "============================================"
echo ""
echo "  1. Open browser: http://localhost:$LABEL_STUDIO_PORT"
echo "  2. Create account (local, no email needed)"
echo "  3. Create new project: 'DSP501 Audio Analysis'"
echo "  4. Settings → Labeling Interface → Code"
echo "     → Paste content from: $LS_DIR/labeling_config.xml"
echo "  5. Settings → Cloud Storage → Add Source Storage"
echo "     → Type: Local files"
echo "     → Path: $LS_DIR/audio_samples"
echo "  6. Import: Upload $LS_DIR/tasks.json"
echo "  7. Start labeling! Click each task to see waveform"
echo ""
echo "  Press Ctrl+C to stop Label Studio"
echo "============================================"
echo ""

"$VENV/label-studio" start --port "$LABEL_STUDIO_PORT"
