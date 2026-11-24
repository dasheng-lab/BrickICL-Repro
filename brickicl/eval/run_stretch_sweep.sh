#!/bin/bash
# Stretch probe, likelihood only: log p(long) - log p(short) for several stretch lengths r and both
# context layouts (scattered = plain BrickNet walk, contiguous = each other copy as one run of lines),
# with frozen BrickNet-PT 0.6B and 1.7B. No sampling, so each (layout, r, model) takes about a minute.
# Usage: bash brickicl/eval/run_stretch_sweep.sh [OUT_DIR]   (default OUT_DIR: runs/stretch_sweep)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
OUT=${1:-$RUNS/stretch_sweep}
GRAPHS=${GRAPHS:-$BRICKNET_DATA/dataset/val.npz.xz}
RS=${RS:-1 2 4 6}
LAYOUTS=${LAYOUTS:-scattered contiguous}
mkdir -p "$OUT"

for layout in $LAYOUTS; do
    for r in $RS; do
        tag="$layout.r$r"
        echo "=== [$(date +%T)] build $tag"
        python -m brickicl.sampling.build_stretch --graphs "$GRAPHS" --out "$OUT/stretch.$tag.jsonl" --r "$r" --layout "$layout"
        for size in 0.6B 1.7B; do
            python -m brickicl.eval.length_probe --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
                --data "$OUT/stretch.$tag.jsonl" --out "$OUT/stretch.$tag.$size.jsonl" --n 0
        done
    done
done
echo "=== [$(date +%T)] report"
for layout in $LAYOUTS; do
    for r in $RS; do
        python -m brickicl.eval.length_probe --report "$OUT/stretch.$layout.r$r.0.6B.jsonl" "$OUT/stretch.$layout.r$r.1.7B.jsonl"
    done
done | tee "$OUT/report.txt"
echo "=== [$(date +%T)] SWEEP DONE"
