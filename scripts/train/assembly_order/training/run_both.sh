#!/usr/bin/env bash
# Train both arms one after the other with identical settings; only --arm differs.
# Run on the GPU machine, detached:
#   nohup bash scripts/train/assembly_order/training/run_both.sh > logs/assembly_order.train_v2.log 2>&1 < /dev/null &
# Extra arguments are passed to both runs (e.g. --lr 5e-5). OUT picks the output directory.
set -euo pipefail
source "$(dirname "$0")/../../../env.sh"
OUT=${OUT:-$RUNS/assembly_order/train_v2}
for arm in masked plain; do
    echo "=== $(date '+%F %T') $arm -> $OUT/$arm"
    python scripts/train/assembly_order/training/train.py --arm "$arm" --output "$OUT/$arm" "$@"
done
echo "=== $(date '+%F %T') both arms done"
