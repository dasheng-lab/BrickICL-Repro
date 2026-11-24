#!/bin/bash
# Two-answer context probes (colour swap, evidence count, evidence position), likelihood only, with frozen
# BrickNet-PT 0.6B and 1.7B. A few minutes per (probe, model).
# Usage: bash brickicl/eval/run_probes.sh [OUT_DIR]   (default OUT_DIR: runs/probes)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
OUT=${1:-$RUNS/probes}
GRAPHS=${GRAPHS:-$BRICKNET_DATA/dataset/val.npz.xz}
PROBES=${PROBES:-color count position}
mkdir -p "$OUT"

for probe in $PROBES; do
    echo "=== [$(date +%T)] build $probe"
    python -m brickicl.sampling.build_probes --probe "$probe" --graphs "$GRAPHS" --out "$OUT/$probe.jsonl"
    for size in 0.6B 1.7B; do
        python -m brickicl.eval.ab_probe --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
            --data "$OUT/$probe.jsonl" --out "$OUT/$probe.$size.jsonl"
    done
done
echo "=== [$(date +%T)] report"
for probe in $PROBES; do
    python -m brickicl.eval.ab_probe --report "$OUT/$probe.0.6B.jsonl" "$OUT/$probe.1.7B.jsonl"
done | tee "$OUT/report.txt"
echo "=== [$(date +%T)] PROBES DONE"
