#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH="$PWD/runs/dfs_posttrain_20261114:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN_DIR=runs/dfs_posttrain_20261114
python -m py_compile "$RUN_DIR/dfs_sampling.py" "$RUN_DIR/build_v2_dfs.py" "$RUN_DIR/build_v3_dfs.py"
if [ ! -f "$RUN_DIR/raw_v2/metadata.json" ]; then
    python -u "$RUN_DIR/build_v2_dfs.py" --out "$RUN_DIR/raw_v2" --walks 8 --workers 8 --include-pt
fi
if [ ! -f "$RUN_DIR/raw_v3/metadata.json" ]; then
    python -u "$RUN_DIR/build_v3_dfs.py" --out "$RUN_DIR/raw_v3" --pt-cache runs/posttrain_icl_v2/pt.cache.npz --walks 6 --workers 8
fi
