#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN_DIR=runs/dfs_curated_20261114
export PYTHONPATH="$PWD/$RUN_DIR:$PWD"
while [ ! -f "$RUN_DIR/DATA_READY.json" ]; do
    if ! pgrep -u "$(id -u)" -f "[p]ython -u runs/dfs_posttrain_20261114/curate_active.py" >/dev/null; then
        echo "Curation stopped before readiness"; exit 1
    fi
    sleep 15
done
python -u runs/dfs_posttrain_20261114/prepare_curated_training.py
for arm in PT_M SFT_M PT_G SFT_G; do
    python -u "$RUN_DIR/train.py" --config "$RUN_DIR/training/$arm/config.json" --resume
done
python -c "import json;from pathlib import Path;p=Path('$RUN_DIR');s={a:json.loads((p/'training'/a/'COMPLETED.json').read_text()) for a in ('PT_M','SFT_M','PT_G','SFT_G')};(p/'COMPLETED.json').write_text(json.dumps(s,indent=2))"
