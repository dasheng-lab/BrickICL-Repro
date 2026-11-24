#!/usr/bin/env bash
# Run the five models on the three shared test sets, with the step 5 completion evaluation.
#
#   val      the step 5 test set (1,208 completion rows from 442 val models); the results of frozen, plain
#            and masked are copied from step 5, only the teammate's model is run
#   longleg  build_longleg.py: val models with every copy of a repeated limb stretched, one copy held out
#   repeat   build_repeat.py: val models with a repeated part group (wheels, lamps, legs ...), nothing edited,
#            one copy held out
#
# Models: frozen (official BrickNet-0.6B-PT), plain and masked (ours, step 4), and the teammate's two
# post-trainings on the official PT, read from their folder (nothing is written there):
#   sunshq        "M only": given G \ M, loss on M alone
#   sunshq_fullg  "full G": generate all of G from nothing
#
# Run on the GPU machine, detached:
#   nohup bash experiments/assembly_order/robustness_evaluation/run_shared.sh > logs/assembly_order.shared.log 2>&1 < /dev/null &
# SETS picks the sets (default: longleg repeat val), MODELS the models (default: all five). A result file that already has every row is kept, so the
# script can be started again.
set -euo pipefail
source "$(dirname "$0")/../../brickicl/env.sh"
HERE=experiments/assembly_order/robustness_evaluation
OUT=${OUT:-$RUNS/assembly_order/eval_shared}
STEP5=${STEP5:-$RUNS/assembly_order/eval_v2}
TRAIN=${TRAIN:-$RUNS/assembly_order/train_v2}
PT=${PT:-kulits/BrickNet-0.6B-PT}
SUNSHQ=${SUNSHQ:-/home/sunshq114514/brickICL/runs/bricknet_06_reference_last/PT/checkpoint-001750}
SUNSHQ_FULLG=${SUNSHQ_FULLG:-/home/sunshq114514/brickICL/runs/bricknet_06_fullg/PT/checkpoint-000500}
ALL="frozen plain masked sunshq sunshq_fullg"
N=${N:-8}

adapters() {  # the --lora arguments of one model
    ADAPTERS=(--lora "$PT")
    case "$1" in
        frozen) ;;
        sunshq) ADAPTERS+=(--lora "$SUNSHQ") ;;
        sunshq_fullg) ADAPTERS+=(--lora "$SUNSHQ_FULLG") ;;
        *) ADAPTERS+=(--lora "$TRAIN/$1/checkpoint-$(python -c "import json; print(json.load(open('$TRAIN/$1/summary.json'))['best_step'])")") ;;
    esac
}

run() {  # run <set> <model>
    local data=$OUT/$1/test.jsonl res=$OUT/$1/$2.jsonl
    if [ -s "$res" ] && [ "$(wc -l < "$res")" = "$(wc -l < "$data")" ]; then
        echo "=== $1 / $2: already done"
        return
    fi
    echo "=== $(date '+%F %T') $1 / $2"
    adapters "$2"
    python experiments/assembly_order/completion_evaluation/eval_completion.py --name "$2" --data "$data" --out "$res" --n "$N" "${ADAPTERS[@]}"
}

mkdir -p "$OUT/val" "$OUT/longleg" "$OUT/repeat"
for f in test frozen plain masked; do cp -n "$STEP5/$f.jsonl" "$OUT/val/"; done
[ -s "$OUT/longleg/test.jsonl" ] || python $HERE/build_longleg.py --out "$OUT/longleg/test.jsonl"
[ -s "$OUT/repeat/test.jsonl" ] || python $HERE/build_repeat.py --out "$OUT/repeat/test.jsonl"

for set in ${SETS:-longleg repeat val}; do
    for name in ${MODELS:-$ALL}; do run "$set" "$name"; done
    files=()
    for name in $ALL; do [ -s "$OUT/$set/$name.jsonl" ] && files+=("$OUT/$set/$name.jsonl"); done
    echo "=== $(date '+%F %T') report: $set (${#files[@]} models)"
    python experiments/assembly_order/completion_evaluation/report.py "${files[@]}" | tee "$OUT/$set/report.md"
done
echo "=== $(date '+%F %T') shared eval done"
