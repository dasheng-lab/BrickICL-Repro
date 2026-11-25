"""Build GRPO completion prompts and sample model continuations.

The input pool contains paired depth-first and subtree-last paths.
"""

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("SRC", str(Path(__file__).resolve().parents[4])))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
EVAL_FREE = SRC / "scripts/train/assembly_order/metrics" / "eval_free.py"
MAKE_TEST = SRC / "scripts/train/assembly_order/completion_evaluation" / "make_test.py"
FROZEN = SRC / "src" / "brickicl" / "eval" / "frozen.py"
EXTERNAL = [EVAL_FREE, MAKE_TEST, FROZEN]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_free = _module("assembly_eval_free", EVAL_FREE)
_make_test = _module("assembly_make_test", MAKE_TEST)
sample_free = _free.sample_free  # (model, tok, rows, n, budget, seed, device, extra_parts) -> [[(text, eos)] per row]
cut = _make_test.cut  # masked pool row -> (prefix, target)

from brickicl.eval.frozen import load  # noqa: E402,F401  (model_name, loras, dtype, device) -> (model, tok)

KEEP = ("source", "sample_index", "nodes", "k", "n_prefix", "jump", "lag", "kind", "copies")


def prompts(pool_dir, rows=None, per_model=None, max_k=20, seed=0, skip_pairs=()):
    """A fixed, seeded selection of masked rows as completion prompts (prefix, target, pair, ...).

    The order is a permutation of the pool with `seed`, so the same arguments always give the same list and any
    shard of it (list[i::S]) is reproducible. Rows with |M| > max_k are skipped (their token cap grows with |M|).
    """
    with open(Path(pool_dir) / "masked.jsonl") as f:
        pool = [json.loads(line) for line in f]
    skip = set(skip_pairs)
    rng = np.random.default_rng(seed)
    per = Counter()
    out = []
    for i in rng.permutation(len(pool)).tolist():
        r = pool[i]
        if "n_prefix" not in r or r["k"] > max_k:  # mixed-in shipped walks have no M
            continue
        pair = f"{r['source']}:{r['sample_index']}"
        if pair in skip or (per_model and per[r["source"]] >= per_model):
            continue
        per[r["source"]] += 1
        prefix, target = cut(r)
        out.append({k: r[k] for k in KEEP if k in r} | {"pair": pair, "prefix": prefix, "target": target})
        if rows and len(out) >= rows:
            break
    return out


def device_and_dtype(device=None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return device, ("bfloat16" if device == "cuda" else "float32")
