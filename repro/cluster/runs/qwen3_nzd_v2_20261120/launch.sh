#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN=runs/qwen3_nzd_v2_20261120
python -m compileall -q "$RUN"
start_worker() {
  local name="$1" executable="$2" script="$3"
  if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then echo "$name already running";return;fi
  nohup "$executable" -u "$RUN/$script" >> "$RUN/$name.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/$name.pid";echo "$name started: $!"
}
start_worker queue "$(command -v python)" run_queue.py
start_worker monitor /home/sunshq114514/brickICL/runs/proposal_report_20261111/env/bin/python monitor.py
