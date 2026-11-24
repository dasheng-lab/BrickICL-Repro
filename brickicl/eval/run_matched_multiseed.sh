#!/usr/bin/env bash
# Multi-walk-seed matched ICL experiment.
#
# The held-out copy is fixed while the serialization walk varies. Every row is
# evaluated twice with the same query: once with the complete-G demonstration
# and once without it. Statistics cluster all walk seeds from one source model.
#
# Full run:
#   bash brickicl/eval/run_matched_multiseed.sh
# Smoke run:
#   SIZES="0.6B" WALK_SEEDS="0,1" N=2 bash brickicl/eval/run_matched_multiseed.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

OUT=${1:-$RUNS/twocopy-multiseed}
STRETCH=${STRETCH:-$RUNS/stretch/stretch.contiguous.r4.jsonl}
WALK_SEEDS=${WALK_SEEDS:-0,1,2}
TARGET_INDEX=${TARGET_INDEX:-0}
SIZES=${SIZES:-"0.6B 1.7B 4B"}
N=${N:-8}
TOKEN_BUDGET=${TOKEN_BUDGET:-180000}

if [ ! -f "$STRETCH" ]; then
    echo "missing stretch input: $STRETCH" >&2
    echo "build it with:" >&2
    echo "  python -m brickicl.sampling.build_stretch --graphs \"$BRICKNET_DATA/dataset/val.npz.xz\" --out \"$STRETCH\" --r 4 --seeds 3 --layout contiguous" >&2
    echo "or set STRETCH=/path/to/stretch.contiguous.r4.jsonl" >&2
    exit 1
fi

SEED_TAG=${WALK_SEEDS//,/-}
DATA="$OUT/twocopy.matched.target${TARGET_INDEX}.seeds${SEED_TAG}.jsonl"
NO_DEMO_DATA="$OUT/twocopy.nodemo.target${TARGET_INDEX}.seeds${SEED_TAG}.jsonl"
mkdir -p "$OUT"

echo "=== [$(date +%T)] build target $TARGET_INDEX, walk seeds $WALK_SEEDS"
python -m brickicl.sampling.build_matched \
    --input "$STRETCH" --output "$DATA" --no-demo-output "$NO_DEMO_DATA" \
    --target-index "$TARGET_INDEX" --walk-seeds "$WALK_SEEDS"

read -r -a SIZE_LIST <<< "$SIZES"
MATCHED_OUTPUTS=()
NO_DEMO_OUTPUTS=()
for size in "${SIZE_LIST[@]}"; do
    matched="$OUT/matched.multiseed.$size.jsonl"
    no_demo="$OUT/nodemo.multiseed.$size.jsonl"
    MATCHED_OUTPUTS+=("$matched")
    NO_DEMO_OUTPUTS+=("$no_demo")

    echo "=== [$(date +%T)] matched demo BrickNet-$size-PT"
    python -m brickicl.eval.eval_matched \
        --data "$DATA" --out "$matched" \
        --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
        --n "$N" --token-budget "$TOKEN_BUDGET"

    echo "=== [$(date +%T)] no-demo control BrickNet-$size-PT"
    python -m brickicl.eval.eval_matched \
        --data "$NO_DEMO_DATA" --out "$no_demo" \
        --model "Qwen/Qwen3-$size" --lora "kulits/BrickNet-$size-PT" \
        --n "$N" --token-budget "$TOKEN_BUDGET"
done

echo "=== [$(date +%T)] clustered multi-seed report"
python -m brickicl.eval.score_matched \
    --old-data "$STRETCH" \
    --matched-data "$DATA" --no-demo-data "$NO_DEMO_DATA" \
    --matched "${MATCHED_OUTPUTS[@]}" --no-demo "${NO_DEMO_OUTPUTS[@]}" \
    --json-output "$OUT/matched.multiseed.metrics.json" \
    > "$OUT/matched.multiseed.summary.txt"

echo "=== [$(date +%T)] done: $OUT"
echo "summary: $OUT/matched.multiseed.summary.txt"
echo "metrics: $OUT/matched.multiseed.metrics.json"
