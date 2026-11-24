#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN=runs/qwen3_fullg_reg_20261119
python -m compileall -q "$RUN"
if [ -f "$RUN/run.pid" ] && kill -0 "$(cat "$RUN/run.pid")" 2>/dev/null; then
  echo "Regularization run already active"
else
  nohup python -u "$RUN/run.py" >> "$RUN/run.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/run.pid"; echo "Training runner $!"
fi
if [ -f "$RUN/evaluation.pid" ] && kill -0 "$(cat "$RUN/evaluation.pid")" 2>/dev/null; then
  echo "Post-training evaluator already active"
else
  nohup python -u "$RUN/evaluate_after.py" >> "$RUN/evaluation.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/evaluation.pid"; echo "Post-training evaluator $!"
fi
if [ -f "$RUN/monitor.pid" ] && kill -0 "$(cat "$RUN/monitor.pid")" 2>/dev/null; then
  echo "Monitor already active"
else
  nohup runs/proposal_report_20261111/env/bin/python -u "$RUN/monitor.py" >> "$RUN/monitor.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/monitor.pid"; echo "Loss monitor $!"
fi
