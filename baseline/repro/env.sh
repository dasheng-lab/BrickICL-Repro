# Shared paths for the repro scripts (sourced, not run). Everything is derived from this file's location,
# so baseline/ can be moved as a whole: baseline/bricknet = upstream code + data, baseline/repro = ours.
REPRO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UPSTREAM=$(cd "$REPRO/../bricknet" && pwd)
BIN=${BRICKNET_BIN:-/opt/anaconda3/envs/bricknet/bin}  # override with your env's bin dir
PY=$BIN/python
GENERATE=$UPSTREAM/BrickNet/scripts/generate.py
export BRICKNET_DATA=$UPSTREAM/data
export HF_HUB_DISABLE_PROGRESS_BARS=1
