#!/usr/bin/env bash
set -euo pipefail
PLAYGROUND_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PLAYGROUND_PYTHON=${BRICKICL_UI_PYTHON:-python}
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BRICKNET_DATA=${BRICKNET_DATA:-$PLAYGROUND_REPO/baseline/bricknet/data}
export TMPDIR=${TMPDIR:-$HOME/.cache/brickicl-gradio/tmp}
export GRADIO_TEMP_DIR=${GRADIO_TEMP_DIR:-$HOME/.cache/brickicl-gradio/files}
mkdir -p "$TMPDIR" "$GRADIO_TEMP_DIR"
cd "$PLAYGROUND_REPO"
exec "$PLAYGROUND_PYTHON" -m brickicl.playground.app "$@"
