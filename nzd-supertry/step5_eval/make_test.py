"""Turn a val pool from step3_data/build_data.py into a fixed completion test set.

Each masked row "G \\ M first, M last" is cut where M starts: the prefix (the n_prefix parts of
G \\ M) is what the model is given, the target (the 2k lines of M) is the reference answer. All
models are evaluated on the same file.

Selection: rows with |M| <= --max-k, at most --per-model rows per model, then a random --rows of
them (fixed --seed). The output derives from the gated dataset: keep it under runs/.

Usage (from the repo root):
  python nzd-supertry/step5_eval/make_test.py runs/nzd-supertry/val_pool_v2 --out runs/nzd-supertry/eval_v2/test.jsonl
"""

import argparse
import io
import json
import lzma
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("BRICKNET_DATA", str(REPO / "baseline/bricknet/data"))


def read_graphs(split):
    from bricknet import load_graphs

    path = Path(os.environ["BRICKNET_DATA"]) / "dataset" / f"{split}.npz.xz"
    return load_graphs(io.BytesIO(lzma.open(path).read()))


def signature(graph):
    """The multiset of (part, colour): equal for copies of one model that sit in two splits."""
    return tuple(sorted(zip(map(int, graph.part_ids), map(int, graph.colors))))


def cut(row):
    """(prefix, target) of a masked row: the text before and from the first line of M."""
    lines = row["path"].splitlines(keepends=True)
    a = 1 + 2 * (row["n_prefix"] - 1)
    assert len(lines) == a + 2 * row["k"], (len(lines), a, row["k"])
    return "".join(lines[:a]), "".join(lines[a:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", help="a build_data.py output directory (val)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--per-model", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=20, help="largest M, in parts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-pool", default="runs/nzd-supertry/sft_train_v2",
                    help="drop val models whose parts and colours equal a model of this training pool ('' to skip)")
    args = ap.parse_args()

    with open(Path(args.pool) / "masked.jsonl") as f:
        pool = [json.loads(line) for line in f]
    assert all(r["split"] == "val" for r in pool), "the test set must come from val"

    twins = set()
    if args.train_pool:
        st = json.load(open(Path(args.train_pool) / "stats.json"))
        train_graphs = read_graphs(st["args"]["split"])
        seen = {signature(train_graphs[s]) for s in st["sources"]}
        val_graphs = read_graphs("val")
        twins = {s for s in {r["source"] for r in pool} if signature(val_graphs[s]) in seen}
        print(f"{len(twins)} val models have the same parts and colours as a training model: excluded")

    rng = np.random.default_rng(args.seed)
    by_model = defaultdict(list)
    for i in rng.permutation(len(pool)).tolist():
        r = pool[i]
        if r["source"] not in twins and r["k"] <= args.max_k and len(by_model[r["source"]]) < args.per_model:
            by_model[r["source"]].append(i)
    cand = sorted(i for v in by_model.values() for i in v)
    pick = sorted(rng.choice(cand, size=min(args.rows, len(cand)), replace=False).tolist())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i in pick:
            r = pool[i]
            prefix, target = cut(r)
            row = {key: r[key] for key in ("source", "sample_index", "nodes", "k", "n_prefix", "jump", "lag")}
            row |= {"pair": f"{r['source']}:{r['sample_index']}", "prefix": prefix, "target": target}
            f.write(json.dumps(row) + "\n")

    rows = [pool[i] for i in pick]
    ks = np.array([r["k"] for r in rows])
    print(f"{len(pool)} pool rows, {len(cand)} candidates -> {len(rows)} test rows from "
          f"{len({r['source'] for r in rows})} models -> {out}")
    print(f"  |M|: mean {ks.mean():.1f}, median {np.median(ks):.0f}, max {ks.max()}; "
          f"off-path attachment: {np.mean([r['jump'] for r in rows]):.1%}")
    print("  |M| histogram:", dict(sorted(Counter(ks.tolist()).items())))


if __name__ == "__main__":
    main()
