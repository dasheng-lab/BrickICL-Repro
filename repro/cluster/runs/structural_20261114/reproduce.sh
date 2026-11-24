#!/usr/bin/env bash
# Usage: bash reproduce.sh /absolute/new-workspace [--check|--train]
set -euo pipefail
if [ $# -lt 1 ]; then echo "Usage: bash reproduce.sh /absolute/new-workspace [--check|--train]"; exit 2; fi
TARGET="$1"
SOURCE=/home/sunshq114514/brickICL
REPRO_PYTHON=/home/sunshq114514/.venvs/brickicl-posttrain/bin/python
case "$TARGET" in /*) ;; *) echo "Use an absolute workspace path"; exit 2;; esac
if [ "$(realpath -m "$TARGET")" = "$(realpath "$SOURCE")" ]; then echo "Use a separate workspace"; exit 2; fi
mkdir -p "$TARGET"
cd "$TARGET"
BASE=runs/dfs_posttrain_20261114
RUN_DIR=runs/structural_20261114
mkdir -p "$BASE" "$RUN_DIR" runs/posttrain_icl_v2 baseline/bricknet
if [ ! -f "$BASE/sources.tgz" ]; then
    cp "$SOURCE/$BASE/sources.tgz" "$BASE/sources.tgz"
    tar -xzf "$BASE/sources.tgz"
    touch brickicl/__init__.py
    ln -s "$SOURCE/baseline/bricknet/data" baseline/bricknet/data
    ln -s "$SOURCE/runs/posttrain_icl_v2/pt.cache.npz" runs/posttrain_icl_v2/pt.cache.npz
fi
for name in dfs_sampling.py build_v2_dfs.py prepare_corpus.py curate_active.py train.py bricknet_runtime.py data.py; do
    cp "$SOURCE/$BASE/$name" "$BASE/$name"
done
cp "$SOURCE/runs/dfs_70k_20261114/caption_rules.py" "$RUN_DIR/caption_rules.py"
for arm in PT_M SFT_M PT_G SFT_G; do
    mkdir -p "$BASE/training/$arm"
    cp "$SOURCE/$BASE/training/$arm/config.json" "$BASE/training/$arm/config.json"
done
for name in factory.py build_structures.py assemble.py prepare_training.py status.py; do
    cp "$SOURCE/$RUN_DIR/$name" "$RUN_DIR/$name"
done
export BRICKNET_DATA="$SOURCE/baseline/bricknet/data"
export PYTHONPATH="$PWD/$RUN_DIR:$PWD/$BASE:$PWD:$PWD/baseline/bricknet/BrickNet/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [ $# -ge 2 ] && [ "$2" = "--check" ]; then
    "$REPRO_PYTHON" -c "import factory,build_structures,assemble,caption_rules;from bricknet import load_catalog;print('Dependencies ready, catalog parts',len(load_catalog().id_to_stem))"
    exit 0
fi
"$REPRO_PYTHON" -c "import caption_rules;from pathlib import Path;caption_rules.ROOT=Path('$RUN_DIR');caption_rules.load_bases()"
for pool in sft pt; do
    "$REPRO_PYTHON" "$RUN_DIR/build_structures.py" --pool "$pool" --workers 8
done
"$REPRO_PYTHON" "$RUN_DIR/assemble.py"
"$REPRO_PYTHON" "$RUN_DIR/prepare_training.py"
if [ $# -ge 2 ] && [ "$2" = "--train" ]; then
    for arm in PT_M SFT_M PT_G SFT_G; do
        "$REPRO_PYTHON" "$RUN_DIR/train.py" --config "$RUN_DIR/training/$arm/config.json" --resume
    done
    "$REPRO_PYTHON" -c "import json;from pathlib import Path;p=Path('$RUN_DIR');r={a:json.loads((p/'training'/a/'COMPLETED.json').read_text()) for a in ('PT_M','SFT_M','PT_G','SFT_G')};(p/'COMPLETED.json').write_text(json.dumps(r,indent=2))"
fi
"$REPRO_PYTHON" "$RUN_DIR/status.py"
