#!/bin/bash
# Stretch probe: hold out one leg (or pencil, rocket, ...) whose twins are shown original / stretched /
# removed, and check whether frozen BrickNet-PT 0.6B and 1.7B complete it long or short.
# Usage: bash brickicl/eval/run_stretch.sh [OUT_DIR]   (default OUT_DIR: runs/stretch)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
OUT=${1:-$RUNS/stretch}
GRAPHS=${GRAPHS:-$BRICKNET_DATA/dataset/val.npz.xz}
R=${R:-2}
N=${N:-32}
LAYOUT=${LAYOUT:-contiguous}
PRIME=${PRIME:-0}  # held-out parts given in the prompt
TAG=$LAYOUT.r$R$([ "$PRIME" -gt 0 ] && echo ".p$PRIME")
mkdir -p "$OUT"

echo "=== [$(date +%T)] build stretch probe ($TAG)"
python -m brickicl.sampling.build_stretch --graphs "$GRAPHS" --out "$OUT/stretch.$LAYOUT.r$R.jsonl" --r "$R" --layout "$LAYOUT"
for size in 0.6B 1.7B; do
    echo "=== [$(date +%T)] frozen BrickNet-$size-PT"
    python -m brickicl.eval.length_probe --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
        --data "$OUT/stretch.$LAYOUT.r$R.jsonl" --out "$OUT/stretch.$TAG.$size.jsonl" --n "$N" --prime "$PRIME"
done
echo "=== [$(date +%T)] report"
python -m brickicl.eval.length_probe --report "$OUT/stretch.$TAG.0.6B.jsonl" "$OUT/stretch.$TAG.1.7B.jsonl" | tee "$OUT/report.$TAG.txt"
echo "=== [$(date +%T)] STRETCH DONE"
