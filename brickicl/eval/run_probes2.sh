#!/bin/bash
# Second round of context probes with frozen BrickNet-PT 0.6B and 1.7B:
#   novel  decoration on every other copy: likelihood on all rows + free samples (seed-0 rows)
#   color  free samples on the colour probe (seed-0 rows): are generated wheels / legs in the context colour?
#   where  anchor of the held-out copy first / last / anywhere: exactly-k sampling, placement accuracy
# Usage: bash brickicl/eval/run_probes2.sh [OUT_DIR]   (default OUT_DIR: runs/probes)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
OUT=${1:-$RUNS/probes}
GRAPHS=${GRAPHS:-$BRICKNET_DATA/dataset/val.npz.xz}
N=${N:-4}
mkdir -p "$OUT"

for probe in novel color; do
    [ -f "$OUT/$probe.jsonl" ] || python -m brickicl.sampling.build_probes --probe "$probe" --graphs "$GRAPHS" --out "$OUT/$probe.jsonl"
    for size in 0.6B 1.7B; do
        echo "=== [$(date +%T)] $probe, BrickNet-$size-PT, likelihood + $N free samples"
        python -m brickicl.eval.ab_probe --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
            --data "$OUT/$probe.jsonl" --out "$OUT/$probe.sampled.$size.jsonl" --n "$N" --sample_seed0
    done
done
echo "=== [$(date +%T)] build where"
python -m brickicl.sampling.build_probes --probe where --graphs "$GRAPHS" --out "$OUT/where.jsonl"
for size in 0.6B 1.7B; do
    echo "=== [$(date +%T)] where, BrickNet-$size-PT"
    python -m brickicl.eval.frozen --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
        --data "$OUT/where.jsonl" --out "$OUT/where.$size.jsonl" --n 16
done
echo "=== [$(date +%T)] report"
{
    python -m brickicl.eval.ab_probe --report "$OUT/novel.sampled.0.6B.jsonl" "$OUT/novel.sampled.1.7B.jsonl" \
        "$OUT/color.sampled.0.6B.jsonl" "$OUT/color.sampled.1.7B.jsonl"
    python -m brickicl.eval.report "$OUT/where.0.6B.jsonl" "$OUT/where.1.7B.jsonl"
} | tee "$OUT/report2.txt"
echo "=== [$(date +%T)] PROBES2 DONE"
