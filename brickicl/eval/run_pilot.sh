#!/bin/bash
# Pilot ICL run: hold out one copy of a repeated pendant (wheel, leg, ...) in val, then run frozen
# BrickNet-PT 0.6B and 1.7B on the evidence/ablated pairs and summarize the gain.
# Usage: bash brickicl/eval/run_pilot.sh [OUT_DIR]   (default OUT_DIR: runs/pilot)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
OUT=${1:-$RUNS/pilot}
GRAPHS=${GRAPHS:-$BRICKNET_DATA/dataset/val.npz.xz}
N=${N:-32}
mkdir -p "$OUT"

echo "=== [$(date +%T)] build pilot set"
python -m brickicl.sampling.build --graphs "$GRAPHS" --out "$OUT/pilot.jsonl"
for size in 0.6B 1.7B; do
    echo "=== [$(date +%T)] frozen BrickNet-$size-PT"
    python -m brickicl.eval.frozen --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
        --data "$OUT/pilot.jsonl" --out "$OUT/pilot.$size.jsonl" --n "$N"
done
echo "=== [$(date +%T)] report"
python -m brickicl.eval.report "$OUT/pilot.0.6B.jsonl" "$OUT/pilot.1.7B.jsonl" | tee "$OUT/report.txt"
echo "=== [$(date +%T)] PILOT DONE"
