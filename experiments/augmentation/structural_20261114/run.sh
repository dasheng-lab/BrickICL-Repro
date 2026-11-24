#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
RUN_DIR=runs/structural_20261114
export PYTHONPATH="$PWD/$RUN_DIR:$PWD/runs/dfs_posttrain_20261114:$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
for pool in sft pt; do
    while [ ! -f "$RUN_DIR/$pool/READY.json" ]; do
        if ! pgrep -u "$(id -u)" -f "[p]ython -u runs/structural_20261114/build_structures.py --pool $pool" >/dev/null; then
            echo "Structure builder $pool stopped before completion"; exit 1
        fi
        sleep 20
    done
done
python -u "$RUN_DIR/assemble.py"
python -u "$RUN_DIR/prepare_training.py"
test -s runs/dfs_70k_20261114/SUPERSEDED.json
for arm in PT_M SFT_M PT_G SFT_G; do
    python -u "$RUN_DIR/train.py" --config "$RUN_DIR/training/$arm/config.json" --resume
done
python -c "import json;from pathlib import Path;p=Path('$RUN_DIR');r={a:json.loads((p/'training'/a/'COMPLETED.json').read_text()) for a in ('PT_M','SFT_M','PT_G','SFT_G')};(p/'COMPLETED.json').write_text(json.dumps(r,indent=2))"
