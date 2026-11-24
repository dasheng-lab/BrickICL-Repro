#!/usr/bin/env bash
# Usage: bash reproduce_70k.sh /absolute/new-workspace [--train|--check]
set -euo pipefail
if [ $# -lt 1 ]; then echo "Usage: bash reproduce_70k.sh /absolute/new-workspace [--train|--check]"; exit 2; fi
TARGET="$1"
SOURCE=/home/sunshq114514/brickICL
REPRO_PYTHON=/home/sunshq114514/.venvs/brickicl-posttrain/bin/python
case "$TARGET" in /*) ;; *) echo "The workspace must be an absolute path"; exit 2;; esac
if [ "$(realpath -m "$TARGET")" = "$(realpath "$SOURCE")" ]; then
    echo "Choose a separate workspace for reproduction"; exit 2
fi
mkdir -p "$TARGET"
cd "$TARGET"
BASE=runs/dfs_posttrain_20261114
FINAL=runs/dfs_70k_20261114
mkdir -p "$BASE" "$FINAL" runs/posttrain_icl_v2 baseline/bricknet
if [ ! -f "$BASE/sources.tgz" ]; then
    cp "$SOURCE/$BASE/sources.tgz" "$BASE/sources.tgz"
    tar -xzf "$BASE/sources.tgz"
    touch brickicl/__init__.py
    ln -s "$SOURCE/baseline/bricknet/data" baseline/bricknet/data
    ln -s "$SOURCE/runs/posttrain_icl_v2/pt.cache.npz" runs/posttrain_icl_v2/pt.cache.npz
fi
for name in dfs_sampling.py build_v2_dfs.py build_v3_dfs.py prepare_corpus.py prepare_training.py audit_official.py curate_active.py; do
    cp "$SOURCE/$BASE/$name" "$BASE/$name"
done
for name in caption_rules.py build_v3_extra.py build_extra.py assemble_70k.py prepare_70k_training.py status.py; do
    cp "$SOURCE/$FINAL/$name" "$FINAL/$name"
done
export BRICKNET_DATA="$SOURCE/baseline/bricknet/data"
export PYTHONPATH="$PWD/$FINAL:$PWD/$BASE:$PWD:$PWD/baseline/bricknet/BrickNet/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [ $# -ge 2 ] && [ "$2" = "--check" ]; then
    "$REPRO_PYTHON" -c "from bricknet import load_catalog;import build_v2_dfs,build_v3_dfs,prepare_corpus,curate_active,caption_rules,build_extra,assemble_70k;print('Imports ready; catalog parts',len(load_catalog().id_to_stem))"
    exit 0
fi
if [ ! -f "$BASE/official_path_audit.json" ]; then
    "$REPRO_PYTHON" "$BASE/audit_official.py" > "$BASE/audit_official.log" 2>&1 &
    AUDIT_PID=$!
else AUDIT_PID=""; fi
if [ ! -f "$BASE/raw_v2/metadata.json" ]; then
    "$REPRO_PYTHON" "$BASE/build_v2_dfs.py" --out "$BASE/raw_v2" --walks 8 --workers 8 --include-pt
fi
if [ ! -f "$BASE/raw_v3/metadata.json" ]; then
    "$REPRO_PYTHON" "$BASE/build_v3_dfs.py" --out "$BASE/raw_v3" --pt-cache runs/posttrain_icl_v2/pt.cache.npz --walks 6 --workers 8
fi
"$REPRO_PYTHON" "$BASE/prepare_corpus.py"
if [ -n "$AUDIT_PID" ]; then wait "$AUDIT_PID"; fi
"$REPRO_PYTHON" "$BASE/prepare_training.py"
"$REPRO_PYTHON" "$BASE/curate_active.py"
"$REPRO_PYTHON" "$FINAL/build_extra.py"
"$REPRO_PYTHON" "$FINAL/assemble_70k.py"
"$REPRO_PYTHON" "$FINAL/prepare_70k_training.py"
if [ $# -ge 2 ] && [ "$2" = "--train" ]; then
    for arm in PT_M SFT_M PT_G SFT_G; do
        "$REPRO_PYTHON" "$FINAL/train.py" --config "$FINAL/training/$arm/config.json" --resume
    done
    "$REPRO_PYTHON" -c "import json;from pathlib import Path;p=Path('$FINAL');r={a:json.loads((p/'training'/a/'COMPLETED.json').read_text()) for a in ('PT_M','SFT_M','PT_G','SFT_G')};(p/'COMPLETED.json').write_text(json.dumps(r,indent=2))"
fi
"$REPRO_PYTHON" "$FINAL/status.py"
