#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN=runs/qwen3_direct_eval_20261119
python -m compileall -q "$RUN"
python "$RUN/prepare.py"
start_worker() {
  local label="$1" script="$2"
  shift 2
  if [ -f "$RUN/$label.pid" ] && kill -0 "$(cat "$RUN/$label.pid")" 2>/dev/null; then echo "$label running"; return; fi
  nohup python -u "$RUN/$script" "$@" >> "$RUN/$label.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/$label.pid"; echo "$label started $!"
}
start_worker inference generate.py
start_worker scoring score.py
start_worker verification verify.py --watch
