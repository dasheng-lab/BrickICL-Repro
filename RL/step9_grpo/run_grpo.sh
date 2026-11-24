#!/usr/bin/env bash
# Step 9: GRPO from the start model. Needs the GPU; ask before starting.
#   ROUND=r1 setsid nohup bash step9_grpo/run_grpo.sh > logs/r1_grpo.log 2>&1 < /dev/null &
# Settings (environment):
#   ROUND      output under runs/<ROUND>/train/<ARM>                  default r1
#   ARM        arm name (step 7 evaluates it like the RFT arms)       default grpo
#   START      adapters of the start model, merged in order           default official PT + round-1 masked
#   MERGED     where the merged start model is kept                    default runs/start_merged/masked
#   POOL       training pool                                           default $NZD_RUNS/sft_train_v2
#   GPU        CUDA device                                             default 0
#   PRESET     r1 (grpo.py's defaults) or v2: F2 reward, -0.1 for a text that does not parse or is empty,
#              fixed loss normaliser, 25% stretched-reference prompts, stretched dev check (README, "v2");
#              v2.1: r1 with the F2 reward and M's first part given in every prompt (sunshq's p1), 400 steps,
#              plus the stretched dev check (measures only)
#              v2.2: v2.1 plus 25% stretched-reference training prompts (built for v2; first part given too)
#   GRPO_ARGS  more arguments for grpo.py (e.g. "--steps 5 --dev-rows 50" for a smoke test)
#   ROUND=v2.1 PRESET=v2.1 setsid nohup bash step9_grpo/run_grpo.sh > logs/v2.1_grpo.log 2>&1 < /dev/null &
# Restartable: an existing run directory with checkpoints is resumed.
set -euo pipefail
ENV_NAME=vllm source "$(dirname "$0")/../env.sh"
ROUND=${ROUND:-r1}
ARM=${ARM:-grpo}
START=${START:-"kulits/BrickNet-0.6B-PT $NZD_RUNS/train_v2/masked/checkpoint-2000"}
MERGED=${MERGED:-$RUNS/start_merged/masked}
POOL=${POOL:-$NZD_RUNS/sft_train_v2}
OUT=$RUNS/$ROUND/train/$ARM
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ ! -s "$MERGED/model.safetensors" ]; then  # the merge uses the bricknet env's loader (CPU)
    lora_args=(); for l in $START; do lora_args+=(--lora "$l"); done
    CUDA_VISIBLE_DEVICES="" conda run -n bricknet --no-capture-output \
        python step9_grpo/merge_start.py --out "$MERGED" "${lora_args[@]}" --check
fi
preset=()
if [ "${PRESET:-r1}" = v2 ]; then  # the stretched prompts are built on the CPU, from sft models only
    for part in dev train; do  # dev first: the training build excludes its models
        if [ ! -s "$RUNS/stretch/$part.jsonl" ]; then
            src=(--rest-of "$POOL" --n-models 4000)
            [ $part = train ] && src=(--pool "$POOL" --extra-models 40000 --exclude "$NZD_RUNS/sft_dev_v2" "$RUNS/stretch/dev.jsonl")
            CUDA_VISIBLE_DEVICES="" conda run -n bricknet --no-capture-output \
                python step9_grpo/build_stretch_prompts.py "${src[@]}" --out "$RUNS/stretch/$part.jsonl"
        fi
    done
    preset=(--beta 2 --fail-reward -0.1 --loss-norm const --norm-tokens 70000
            --stretch "$RUNS/stretch/train.jsonl" --stretch-frac 0.25 --stretch-dev "$RUNS/stretch/dev.jsonl")
elif [ "${PRESET:-r1}" = v2.1 ]; then  # r1 with the F2 reward, first part of M given (sunshq's p1) everywhere
    preset=(--beta 2 --first-part --steps 400 --stretch-dev "$RUNS/stretch/dev.jsonl")  # the dev check only measures
elif [ "${PRESET:-r1}" = v2.2 ]; then  # v2.1 plus 25% stretched-reference training prompts (first part given too)
    preset=(--beta 2 --first-part --steps 400 --stretch "$RUNS/stretch/train.jsonl" --stretch-frac 0.25
            --stretch-dev "$RUNS/stretch/dev.jsonl")
elif [ "${PRESET:-r1}" != r1 ]; then
    echo "unknown PRESET=$PRESET" >&2; exit 1
fi
resume=(); ls -d "$OUT"/checkpoint-* > /dev/null 2>&1 && resume=(--resume)
echo "=== $(date '+%F %T') GRPO $ARM -> $OUT"
python step9_grpo/grpo.py --start "$MERGED" --pool "$POOL" --dev "$RUNS/dev_test/test.jsonl" --out "$OUT" \
    "${preset[@]}" "${resume[@]}" ${GRPO_ARGS:-}
echo "=== $(date '+%F %T') GRPO done: $OUT"
