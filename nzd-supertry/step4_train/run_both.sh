#!/usr/bin/env bash
# Train both arms one after the other with identical settings; only --arm differs.
# Run on the GPU machine, detached:
#   nohup bash nzd-supertry/step4_train/run_both.sh > logs/nzd-supertry.train_v2.log 2>&1 < /dev/null &
# Extra arguments are passed to both runs (e.g. --lr 5e-5). OUT picks the output directory.
set -euo pipefail
source "$(dirname "$0")/../../brickicl/env.sh"
OUT=${OUT:-$RUNS/nzd-supertry/train_v2}
for arm in masked plain; do
    echo "=== $(date '+%F %T') $arm -> $OUT/$arm"
    python nzd-supertry/step4_train/train.py --arm "$arm" --output "$OUT/$arm" "$@"
done
echo "=== $(date '+%F %T') both arms done"
