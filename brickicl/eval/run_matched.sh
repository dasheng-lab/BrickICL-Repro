#!/usr/bin/env bash
# One walk seed per source: full G, EOS, matching G/M; eight free samples per condition.
# Usage: bash brickicl/eval/run_matched.sh [OUT_DIR]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
OUT=${1:-$RUNS/twocopy}
STRETCH=${STRETCH:-$RUNS/stretch/stretch.contiguous.r4.jsonl}
N=${N:-8}
mkdir -p "$OUT"

python -m brickicl.sampling.build_matched --input "$STRETCH" \
  --output "$OUT/twocopy.matched.r4.p1.jsonl" \
  --no-demo-output "$OUT/twocopy.matched.nodemo.r4.p1.jsonl"

for size in 0.6B 1.7B 4B; do
  echo "=== [$(date +%T)] matched BrickNet-$size-PT"
  python -m brickicl.eval.eval_matched \
    --data "$OUT/twocopy.matched.r4.p1.jsonl" \
    --out "$OUT/matched.$size.jsonl" \
    --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
    --n "$N" --token-budget 180000
  python -m brickicl.eval.length_probe --report "$OUT/matched.$size.jsonl" \
    > "$OUT/matched.$size.report.txt"
done

echo "=== [$(date +%T)] no-demonstration 4B control"
python -m brickicl.eval.eval_matched \
  --data "$OUT/twocopy.matched.nodemo.r4.p1.jsonl" \
  --out "$OUT/matched.nodemo.4B.jsonl" \
  --model Qwen/Qwen3-4B --lora kulits/BrickNet-4B-PT \
  --n "$N" --token-budget 180000

python -m brickicl.eval.score_matched \
  --old-data "$STRETCH" \
  --matched-data "$OUT/twocopy.matched.r4.p1.jsonl" \
  --no-demo-data "$OUT/twocopy.matched.nodemo.r4.p1.jsonl" \
  --matched "$OUT/matched.0.6B.jsonl" "$OUT/matched.1.7B.jsonl" "$OUT/matched.4B.jsonl" \
  --no-demo "$OUT/matched.nodemo.4B.jsonl" \
  --json-output "$OUT/matched.r4.p1.metrics.json" \
  > "$OUT/matched.r4.p1.summary.txt"
echo "=== [$(date +%T)] done: $OUT"
