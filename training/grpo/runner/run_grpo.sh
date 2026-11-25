#!/usr/bin/env bash
# GRPO on the supervised completion checkpoint. Run from the repository root.
# Set START (ordered base adapters), POOL, and DEV for your data.
set -euo pipefail
ENV_NAME=vllm source "$(dirname "$0")/../env.sh"

ROUND=${ROUND:-grpo}
START=${START:-"kulits/BrickNet-0.6B-PT $ORDER_RUNS/train_v2/masked/checkpoint-2000"}
POOL=${POOL:-$ORDER_RUNS/sft_train_v2}
DEV=${DEV:-$RUNS/dev_test/test.jsonl}
MERGED=${MERGED:-$RUNS/start_merged/masked}
OUT=${OUT:-$RUNS/$ROUND/train/grpo}
case "$POOL" in /*) ;; *) POOL="$SRC/$POOL" ;; esac
case "$DEV" in /*) ;; *) DEV="$SRC/$DEV" ;; esac
case "$MERGED" in /*) ;; *) MERGED="$SRC/$MERGED" ;; esac
case "$OUT" in /*) ;; *) OUT="$SRC/$OUT" ;; esac
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

test -f "$POOL/masked.jsonl" || { echo "Missing training pool: $POOL/masked.jsonl" >&2; exit 2; }
test -f "$DEV" || { echo "Missing development prompts: $DEV" >&2; exit 2; }
if [ ! -s "$MERGED/model.safetensors" ]; then
    lora_args=()
    for adapter in $START; do
        if [ -e "$SRC/$adapter" ]; then adapter="$SRC/$adapter"; fi
        lora_args+=(--lora "$adapter")
    done
    CUDA_VISIBLE_DEVICES="" conda run -n bricknet --no-capture-output \
        python runner/merge_start.py --out "$MERGED" "${lora_args[@]}" --check
fi

resume=()
if compgen -G "$OUT/checkpoint-*" >/dev/null; then resume=(--resume); fi
python runner/grpo.py --start "$MERGED" --pool "$POOL" --dev "$DEV" --out "$OUT" \
    --beta 2 --first-part --steps 400 "${resume[@]}"
