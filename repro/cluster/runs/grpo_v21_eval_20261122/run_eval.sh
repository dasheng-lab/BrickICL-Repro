#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN=runs/grpo_v21_eval_20261122
python "$RUN/prepare.py"
python -u "$RUN/generate.py"
python -u "$RUN/score.py"
python "$RUN/verify.py"
python runs/viewer_parse_fox_fix_20261121/future_patch.py "$RUN"
python "$RUN/verify.py"
python runs/no_eos_eval_20261120/evaluate_future.py "$RUN"
