#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN=runs/no_eos_eval_20261120
python -m compileall -q "$RUN"
python "$RUN/prepare.py"
start_worker() {
  local name="$1" script="$2"
  if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then echo "$name already active";return;fi
  nohup python -u "$RUN/$script" >> "$RUN/$name.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/$name.pid";echo "$name started $!"
}
start_worker inference generate.py
start_worker scoring score.py
