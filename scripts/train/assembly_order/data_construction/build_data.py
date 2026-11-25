"""Build the paired training pools: depth-first sequences of the same tree, with and without M moved last.

Writes three files into --out:
  masked.jsonl  "G \\ M first, M last": the preorder with the block of M moved to the end (augmentation arm)
  plain.jsonl   the same tree in depth-first preorder, the order BrickNet was trained on (control arm);
                row i pairs with masked row i
  stats.json    what was used, skipped and dropped

Rows follow the shipped `paths_*.jsonl` (`source`, `round`, `nodes`, `path`), so upstream
`third_party/bricknet/BrickNet/scripts/train.py pt --paths masked.jsonl` reads them as they are. Outputs derive from the gated
dataset: keep them under runs/ (not tracked).

Usage (from the repo root, in the bricknet env):
  python scripts/train/assembly_order/data_construction/build_data.py --split sft --models 2000 --out runs/assembly_order/sft_2k
"""

import argparse
import hashlib
import io
import json
import lzma
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("BRICKNET_DATA", str(REPO / "data/bricknet"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sampling"))

from bricknet import load_graphs  # noqa: E402
from sampler import n_pairs_for, sample_pairs, serializable  # noqa: E402

GRAPHS = None  # set in the parent before the pool forks, shared copy-on-write
ARGS = None


def read_graphs(split):
    path = Path(os.environ["BRICKNET_DATA"]) / "dataset" / f"{split}.npz.xz"
    return load_graphs(io.BytesIO(lzma.open(path).read()))


def skip_reason(g, args):
    if len(g.components) != 1:
        return "multi_component"
    n = len(g.part_ids)
    if n < args.min_parts:
        return "too_small"
    if n > args.max_model_parts:
        return "too_large"
    if not serializable(g):
        return "unknown_colour"
    return None


def work(gid):
    g = GRAPHS[gid]
    n = min(len(g.part_ids), ARGS.max_parts)
    pairs, stats = sample_pairs(
        g,
        n_pairs_for(n, ARGS.parts_per_M, ARGS.max_M),
        seed=ARGS.seed * 1_000_003 + gid,
        max_parts=ARGS.max_parts,
        max_frac=ARGS.max_frac,
        min_parts=ARGS.min_parts,
    )
    return gid, pairs, stats


def main():
    global GRAPHS, ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["pt", "sft", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", type=int, default=0, help="random subset of the usable models (0: all)")
    ap.add_argument("--offset", type=int, default=0, help="skip this many models of the shuffle first (for a disjoint dev set)")
    ap.add_argument("--min-parts", type=int, default=6)
    ap.add_argument("--max-model-parts", type=int, default=100, help="skip larger models (their walks are sub-models)")
    ap.add_argument("--max-parts", type=int, default=100, help="parts per sequence, as in BrickNet")
    ap.add_argument("--parts-per-M", type=float, default=4, help="draw about one M per this many parts")
    ap.add_argument("--max-M", type=int, default=25)
    ap.add_argument("--max-frac", type=float, default=0.5, help="largest M as a fraction of the sequence")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ARGS = ap.parse_args()

    t0 = time.time()
    GRAPHS = read_graphs(ARGS.split)
    skipped = Counter()
    usable = []
    for gid, g in enumerate(GRAPHS):
        why = skip_reason(g, ARGS)
        if why:
            skipped[why] += 1
        else:
            usable.append(gid)
    if ARGS.models or ARGS.offset:
        # one fixed shuffle per seed; disjoint slices of it give disjoint model sets (train / dev)
        perm = np.random.default_rng(ARGS.seed).permutation(len(usable))
        pick = perm[ARGS.offset : ARGS.offset + ARGS.models] if ARGS.models else perm[ARGS.offset :]
        usable = [usable[i] for i in sorted(pick.tolist())]
    print(f"{len(GRAPHS)} graphs in {ARGS.split}, skipped {dict(skipped)}, sampling {len(usable)} ({time.time() - t0:.0f} s)")

    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    stats, k_hist = Counter(), Counter()
    lag = {"plain": [], "masked": []}
    seen = (set(), set())  # md5 of every masked / plain text written so far
    n_rows = n_models = 0
    t0 = time.time()
    # fork: the workers inherit GRAPHS; the pybullet client is created lazily, after the fork
    with mp.get_context("fork").Pool(ARGS.workers) as pool, open(out / "masked.jsonl", "w") as fm, open(
        out / "plain.jsonl", "w"
    ) as fp:
        for gid, pairs, st in pool.imap(work, usable, chunksize=16):
            stats.update(st)
            n_models += bool(pairs)
            for j, pair in enumerate(pairs):
                # identical models exist in the dataset; like the shipped pools, keep each text once
                digests = [hashlib.md5(pair[arm]["path"].encode()).digest() for arm in ("masked", "plain")]
                if digests[0] in seen[0] or digests[1] in seen[1]:
                    stats["duplicate_of_other_model"] += 1
                    continue
                seen[0].add(digests[0])
                seen[1].add(digests[1])
                for arm, f in (("masked", fm), ("plain", fp)):
                    row = {"source": gid, "split": ARGS.split, "round": f"paired-{arm}-v2", "sample_index": j}
                    row.update(pair[arm])
                    f.write(json.dumps(row) + "\n")
                    lag[arm].append(row["lag"])
                k_hist[pair["masked"]["k"]] += 1
                n_rows += 1
    dt = time.time() - t0

    ks = np.array(sorted(k_hist.elements()))
    summary = {
        "args": vars(ARGS),
        "graphs": len(GRAPHS),
        "skipped": dict(skipped),
        "models_sampled": len(usable),
        "sources": usable,
        "models_with_pairs": n_models,
        "pairs": n_rows,
        "sampler": dict(stats),
        "k": {"mean": float(ks.mean()), "median": float(np.median(ks)), "p90": float(np.percentile(ks, 90)), "max": int(ks.max())},
        # distance, in parts, from the first part of M back to the part it attaches to (1: the previous part)
        "attach_lag_mean": {a: float(np.mean(v)) for a, v in lag.items()},
        "attach_lag_median": {a: float(np.median(v)) for a, v in lag.items()},
        # share of masked rows whose M attaches off the current root-to-last path (never seen in a preorder)
        "jump_share": stats["jump"] / max(stats["jump"] + stats["no_jump"], 1),
        "seconds": round(dt, 1),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{n_rows} pairs from {n_models} models in {dt:.0f} s ({ARGS.workers} workers) -> {out}")
    print(f"  sampler: {dict(stats)}")
    print(f"  |M|: mean {ks.mean():.1f}, median {np.median(ks):.0f}, p90 {np.percentile(ks, 90):.0f}, max {ks.max()}")
    print(f"  masked rows with an off-path attachment: {summary['jump_share']:.1%}")
    print(f"  attach lag (parts back to the anchor), mean / median: "
          f"masked {summary['attach_lag_mean']['masked']:.1f} / {summary['attach_lag_median']['masked']:.0f}, "
          f"plain {summary['attach_lag_mean']['plain']:.1f} / {summary['attach_lag_median']['plain']:.0f}")


if __name__ == "__main__":
    main()
