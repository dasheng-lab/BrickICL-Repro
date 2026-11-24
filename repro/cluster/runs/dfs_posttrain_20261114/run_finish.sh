#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
RUN_DIR=runs/dfs_posttrain_20261114
export PYTHONPATH="$PWD/$RUN_DIR:$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
while [ ! -f "$RUN_DIR/raw_v3/metadata.json" ]; do
    if ! pgrep -u "$(id -u)" -f "[b]ash runs/dfs_posttrain_20261114/run_build.sh" >/dev/null; then
        echo "Data builder stopped before completion"; exit 1
    fi
    sleep 30
done
python -u "$RUN_DIR/prepare_corpus.py"
while [ ! -f "$RUN_DIR/official_path_audit.json" ]; do
    if ! pgrep -u "$(id -u)" -f "[p]ython3 -u runs/dfs_posttrain_20261114/audit_official.py" >/dev/null; then
        echo "Official path audit stopped before completion"; exit 1
    fi
    sleep 30
done
python -u "$RUN_DIR/prepare_training.py"
for arm in PT_M SFT_M PT_G SFT_G; do
    python -u "$RUN_DIR/train.py" --config "$RUN_DIR/training/$arm/config.json" --resume
done
python -c "import json;from pathlib import Path;p=Path('$RUN_DIR');s={a:json.loads((p/'training'/a/'COMPLETED.json').read_text()) for a in ('PT_M','SFT_M','PT_G','SFT_G')};(p/'COMPLETED.json').write_text(json.dumps(s,indent=2));print('ALL FOUR TRAINING RUNS COMPLETED')"
