#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN_DIR=runs/structural_eval_20261115
while [ ! -f "$RUN_DIR/EXTRA_COMPLETED.json" ]; do sleep 20; done
python -u "$RUN_DIR/pack_viewer.py"
python -u "$RUN_DIR/check_registry.py"
runs/proposal_report_20261111/env/bin/python "$RUN_DIR/verify_viewer.py"
