#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN=runs/qwen3_direct_pt_20261117
start_worker() {
  local label="$1" executable="$2" script="$3"
  if [ -f "$RUN/$label.pid" ] && kill -0 "$(cat "$RUN/$label.pid")" 2>/dev/null; then
    echo "$label already running"; return
  fi
  nohup "$executable" -u "$RUN/$script" >> "$RUN/$label.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/$label.pid"
  echo "$label started: $!"
}
python -m compileall -q "$RUN"
start_worker official_build "$(command -v python)" build_official.py
if [ ! -f "$RUN/BASE_PROVENANCE.json" ]; then
  python -u "$RUN/restore_base.py"
fi
start_worker queue "$(command -v python)" run_queue.py
start_worker monitor /home/sunshq114514/brickICL/runs/proposal_report_20261111/env/bin/python monitor.py
