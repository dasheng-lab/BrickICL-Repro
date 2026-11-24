#!/usr/bin/env bash
# Step 5: the completion test on val for the three models, then the comparison tables.
# Run on the GPU machine, detached:
#   nohup bash nzd-supertry/step5_eval/run_eval.sh > logs/nzd-supertry.eval_v2.log 2>&1 < /dev/null &
# TRAIN: directory with the two training runs; OUT: where results go; N: samples per row.
# STEP: checkpoint step for both arms (default: each arm's lowest dev loss, from its summary.json).
set -euo pipefail
source "$(dirname "$0")/../../brickicl/env.sh"
HERE=nzd-supertry/step5_eval
TRAIN=${TRAIN:-$RUNS/nzd-supertry/train_v2}
OUT=${OUT:-$RUNS/nzd-supertry/eval_v2}
POOL=${POOL:-$RUNS/nzd-supertry/val_pool_v2}
PT=${PT:-kulits/BrickNet-0.6B-PT}
N=${N:-8}
mkdir -p "$OUT"

adapters() {  # the --lora arguments of one model
    ADAPTERS=(--lora "$PT")
    [ "$1" = frozen ] && return
    local step=${STEP:-$(python -c "import json; print(json.load(open('$TRAIN/$1/summary.json'))['best_step'])")}
    ADAPTERS+=(--lora "$TRAIN/$1/checkpoint-$step")
}

echo "=== $(date '+%F %T') test set"
[ -f "$POOL/masked.jsonl" ] || python nzd-supertry/step3_data/build_data.py --split val --seed 0 --out "$POOL"
python $HERE/make_test.py "$POOL" --out "$OUT/test.jsonl"

for name in frozen plain masked; do
    echo "=== $(date '+%F %T') completion: $name"
    adapters "$name"
    echo "    ${ADAPTERS[*]}"
    python $HERE/eval_completion.py --name "$name" --data "$OUT/test.jsonl" --out "$OUT/$name.jsonl" \
        --n "$N" "${ADAPTERS[@]}"
done

echo "=== $(date '+%F %T') report"
python $HERE/report.py "$OUT"/{frozen,plain,masked}.jsonl | tee "$OUT/report.md"

# Has the post-training cost anything on the order BrickNet was trained on? Loss on the shipped val walks
# (and, for reference, on whole sequences of the val pool in both orders).
for name in frozen plain masked; do
    echo "=== $(date '+%F %T') retention: $name"
    adapters "$name"
    python nzd-supertry/step4_train/check_order_loss.py orders val "${ADAPTERS[@]}" | tee "$OUT/orders.$name.txt"
    python nzd-supertry/step4_train/check_order_loss.py pool "$POOL" "${ADAPTERS[@]}" | tee "$OUT/pool.$name.txt"
done
echo "=== $(date '+%F %T') eval done"
