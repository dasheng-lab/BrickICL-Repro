#!/usr/bin/env bash
# Compare three matched long/short-leg contexts on the same source/walk units.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

OUT=${1:-$RUNS/context-comparison}
STRETCH=${STRETCH:-$RUNS/stretch/stretch.contiguous.r4.jsonl}
CAPTIONS=${CAPTIONS:-$BRICKNET_DATA/dataset/captions_val.jsonl.xz}
WALK_SEEDS=${WALK_SEEDS:-0,1,2}
TARGET_INDEX=${TARGET_INDEX:-0}
SIZES=${SIZES:-"1.7B"}
N=${N:-8}
TOKEN_BUDGET=${TOKEN_BUDGET:-180000}

if [ ! -f "$STRETCH" ]; then
    echo "missing stretch input: $STRETCH" >&2
    exit 1
fi
if [ ! -f "$CAPTIONS" ]; then
    echo "missing validation captions: $CAPTIONS" >&2
    exit 1
fi

SEED_TAG=${WALK_SEEDS//,/-}
MATCHED_DATA="$OUT/double.target${TARGET_INDEX}.seeds${SEED_TAG}.jsonl"
SINGLE_DATA="$OUT/single.target${TARGET_INDEX}.seeds${SEED_TAG}.jsonl"
SFT_DATA="$OUT/sft-caption-partial.target${TARGET_INDEX}.seeds${SEED_TAG}.jsonl"
mkdir -p "$OUT"

echo "=== [$(date +%T)] build paired context inputs"
python -m brickicl.sampling.build_matched \
    --input "$STRETCH" \
    --output "$MATCHED_DATA" \
    --no-demo-output "$SINGLE_DATA" \
    --sft-output "$SFT_DATA" \
    --captions "$CAPTIONS" \
    --target-index "$TARGET_INDEX" \
    --walk-seeds "$WALK_SEEDS"

read -r -a SIZE_LIST <<< "$SIZES"
MATCHED_OUTPUTS=()
SINGLE_OUTPUTS=()
SFT_OUTPUTS=()
for size in "${SIZE_LIST[@]}"; do
    model="Qwen/Qwen3-$size"
    pt="kulits/BrickNet-$size-PT"
    sft="kulits/BrickNet-$size-SFT"
    matched="$OUT/double.$size.jsonl"
    single="$OUT/single.$size.jsonl"
    sft_out="$OUT/sft-caption-partial.$size.jsonl"
    MATCHED_OUTPUTS+=("$matched")
    SINGLE_OUTPUTS+=("$single")
    SFT_OUTPUTS+=("$sft_out")

    echo "=== [$(date +%T)] PT single-copy BrickNet-$size-PT"
    python -m brickicl.eval.eval_matched \
        --data "$SINGLE_DATA" --out "$single" \
        --model "$model" --lora "$pt" \
        --n "$N" --token-budget "$TOKEN_BUDGET"

    echo "=== [$(date +%T)] PT double-copy BrickNet-$size-PT"
    python -m brickicl.eval.eval_matched \
        --data "$MATCHED_DATA" --out "$matched" \
        --model "$model" --lora "$pt" \
        --n "$N" --token-budget "$TOKEN_BUDGET"

    echo "=== [$(date +%T)] PT+SFT caption + partial path BrickNet-$size-SFT"
    python -m brickicl.eval.eval_matched \
        --data "$SFT_DATA" --out "$sft_out" \
        --model "$model" --lora "$pt" --lora "$sft" \
        --n "$N" --token-budget "$TOKEN_BUDGET"
done

echo "=== [$(date +%T)] paired three-context report"
python -m brickicl.eval.score_matched \
    --old-data "$STRETCH" \
    --matched-data "$MATCHED_DATA" --matched "${MATCHED_OUTPUTS[@]}" \
    --no-demo-data "$SINGLE_DATA" --no-demo "${SINGLE_OUTPUTS[@]}" \
    --sft-data "$SFT_DATA" --sft-context "${SFT_OUTPUTS[@]}" \
    --json-output "$OUT/context-comparison.metrics.json" \
    > "$OUT/context-comparison.summary.txt"

echo "=== [$(date +%T)] done: $OUT"
echo "summary: $OUT/context-comparison.summary.txt"
echo "metrics: $OUT/context-comparison.metrics.json"
