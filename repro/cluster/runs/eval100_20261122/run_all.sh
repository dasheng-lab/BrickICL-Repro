#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
umask 002
python -u runs/eval100_20261122/run_eval.py
python -u runs/eval100_20261122/publish_results.py
python runs/viewer_parse_fox_fix_20261121/restart_ui.py
runs/proposal_report_20261111/env/bin/python runs/eval100_20261122/verify_published.py
python runs/eval100_20261122/export_report.py
