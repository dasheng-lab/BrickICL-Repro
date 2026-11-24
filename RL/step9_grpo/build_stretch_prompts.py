"""Stretched-reference completion prompts for GRPO v2, from sft models only. (CPU, bricknet env)

r1 never saw a prompt where the reference asks for something the prior does not, so nothing rewarded following
it (sunshq's sampled Δlog-odds fell from +2.7 to +1.8 with the first part given). These prompts are built the way
nzd's step 8 / step 11 long-leg rows are (build_longleg.held_out_row and brickicl.sampling.edits.stretch, imported
read only): a model with a repeated, stretchable limb, one copy held out as M, another copy last in the prefix.
Every copy is written in one of two conditions, always as a pair, so the length of M can only be read off the
reference:

  stretched  every copy r stacked parts longer; M = the long held-out copy
  original   the model as in the dataset;       M = the copy as built

(Unlike step 11's "original" condition, M agrees with its context here: both rows are correct completions.)
r is drawn per copy from --r (default 1-5), so the model has to count, not learn "+4".

Sources, all from sft.npz.xz (val, where the step 11 / sunshq long-leg tests live, is never read). Only about 2-4%
of models have a stretchable limb, so the pools alone are small (sft_train_v2: 473 pairs from 243 models;
sft_dev_v2: about 12 pairs):
  dev check  --rest-of sft_train_v2 --n-models 4000: sft models drawn (seeded) from those outside the training pool
  training   --pool sft_train_v2 --extra-models N --exclude sft_dev_v2 <dev build>: the pool plus N other sft
             models, none of them in the dev test's pool or in the dev check Each row has the fields of a step 5 test row
plus kind "stretch", condition, r, and the extra part (extra_part, as brickicl.eval.match.realize names it;
n_extra = its count in this row's M, n_extra_long = its count in the long M), so the trainer can report the
long-feature hit rate (generated count of the extra part >= n_extra_long, sunshq's rule) per condition.

Usage (from ~/RL, `source env.sh`):
  python step9_grpo/build_stretch_prompts.py --rest-of $NZD_RUNS/sft_train_v2 --n-models 4000 --out runs/stretch/dev.jsonl
  python step9_grpo/build_stretch_prompts.py --pool $NZD_RUNS/sft_train_v2 --extra-models 40000 \\
      --exclude $NZD_RUNS/sft_dev_v2 runs/stretch/dev.jsonl --out runs/stretch/train.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("SRC", "/home/nzd0420/brickICL"))
NZD = SRC / "nzd-supertry"
for p in (SRC, NZD / "step2_sampler", NZD / "step8_shared_test"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARGS = None
GRAPHS = None


def init(args):
    global ARGS, GRAPHS
    from brickicl.sampling.build import read_graphs

    ARGS = args
    GRAPHS = read_graphs(args.graphs)


def guarded_rows_of(gid):
    """rows_of, with a model that nzd's code cannot handle (an unknown part id, ...) skipped and counted."""
    try:
        return rows_of(gid)
    except Exception as e:  # noqa: BLE001
        return [], Counter({f"model failed: {type(e).__name__}": 1})


def rows_of(gid):
    """Both conditions for up to --per-model held-out copies of one model; ([rows], Counter of skips)."""
    from build_longleg import dataset_collisions, held_out_row

    from brickicl.eval.match import realize
    from brickicl.sampling.build_stretch import stretchable_group
    from brickicl.sampling.edits import stretch

    a, g, skipped, out = ARGS, GRAPHS[gid], Counter(), []
    try:
        grp, joints = stretchable_group(g, a.max_size, 2)
    except TypeError:  # missing.pendant_copies sorts edge labels; a few sft models have a None label
        return out, skipped | Counter({"pendant search failed (None edge label)": 1})
    if grp is None:
        return out, skipped | Counter({"no stretchable limb": 1})
    n = len(grp.copies)
    base = None
    for t in range(min(n, a.per_model)):
        rng = np.random.default_rng([a.seed, gid, t])
        r = int(rng.choice(a.r))
        if len(g.part_ids) + r * n > a.max_parts:
            skipped["over max parts once stretched"] += 1
            continue
        if len(grp.copies[t]) + r > a.max_k:
            skipped["long M over max k"] += 1
            continue
        base = dataset_collisions(g, a.max_parts) if base is None else base
        g_all, added = stretch(g, joints, r)
        conditions = {
            "stretched": (g_all, list(grp.copies[t]) + added[t], [list(c) + x for j, (c, x) in
                                                                   enumerate(zip(grp.copies, added)) if j != t]),
            "original": (g, list(grp.copies[t]), [list(c) for j, c in enumerate(grp.copies) if j != t]),
        }
        pair = {}
        for cond, (graph, M, others) in conditions.items():
            try:
                row, why = held_out_row(graph, M, others, base, rng, a.tries, a.max_parts)
            except AssertionError as e:  # held_out_row checks its own text; a model it cannot write is skipped
                row, why = None, f"held_out_row check failed ({str(e)[:40]})"
            if row is None:
                skipped[f"{cond}: {why}"] += 1
                break
            pair[cond] = row
        if len(pair) < 2:
            continue
        counts = {}
        for cond, row in pair.items():
            parts, _, err = realize(row["prefix"] + row["target"])
            assert err is None, err
            counts[cond] = Counter(str(p.part_id) for p in parts[row["n_prefix"]:])
        grown = counts["stretched"] - counts["original"]
        if len(grown) != 1 or next(iter(grown.values())) != r:
            skipped["extra part not identified"] += 1
            continue
        extra = next(iter(grown))
        for cond, row in pair.items():
            out.append({"source": gid, "sample_index": t, "pair": f"stretch:{gid}:{t}:{cond}", "kind": "stretch",
                        "condition": cond, "r": r, "copies": n, "k_short": len(grp.copies[t]),
                        "k_long": len(grp.copies[t]) + r, "extra_part": extra, "n_extra": counts[cond][extra],
                        "n_extra_long": counts["stretched"][extra], **row})
    return out, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="a step 3 sft pool; its models are the sources")
    src.add_argument("--rest-of", help="a step 3 sft pool; sources are --n-models other sft models")
    ap.add_argument("--n-models", type=int, default=4000, help="--rest-of: how many other sft models")
    ap.add_argument("--extra-models", type=int, default=0, help="--pool: add this many sft models outside it")
    ap.add_argument("--exclude", nargs="*", default=[], help="pool folders or earlier builds whose models to skip")
    ap.add_argument("--graphs", default=str(SRC / "baseline/bricknet/data/dataset/sft.npz.xz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="stacked parts added, drawn per copy")
    ap.add_argument("--max-size", type=int, default=12, help="largest copy before stretching (as step 8)")
    ap.add_argument("--max-k", type=int, default=14, help="largest M (the long copy)")
    ap.add_argument("--max-parts", type=int, default=100)
    ap.add_argument("--per-model", type=int, default=2)
    ap.add_argument("--tries", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="first N source models only (smoke test)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from common.files import write_jsonl, write_run_config

    from brickicl.sampling.build import read_graphs

    pool_dir = Path(args.pool or args.rest_of)
    stats = json.load(open(pool_dir / "stats.json"))
    assert stats["args"]["split"] == "sft" and "sft" in Path(args.graphs).name, "sft models only (never val)"
    with open(pool_dir / "masked.jsonl") as f:
        in_pool = sorted({json.loads(line)["source"] for line in f})
    excluded = set()
    for e in args.exclude:  # a pool folder (its models) or a jsonl with a "source" field (an earlier build)
        e = Path(e)
        with open(e / "masked.jsonl" if e.is_dir() else e) as f:
            excluded |= {json.loads(line)["source"] for line in f if line.strip()}
    rest = sorted(set(range(len(read_graphs(args.graphs)))) - set(in_pool) - excluded)
    n_more = args.n_models if args.rest_of else args.extra_models
    more = np.random.default_rng(args.seed).choice(rest, n_more, replace=False).tolist() if n_more else []
    base = [] if args.rest_of else [s for s in in_pool if s not in excluded]
    sources = sorted(base + more)[: args.limit]
    print(f"{len(base)} pool models + {len(more)} other sft models ({len(excluded)} excluded)", flush=True)
    rows, skipped = [], Counter()
    with get_context("spawn").Pool(args.workers, initializer=init, initargs=(args,)) as pool:
        for r, s in pool.imap(guarded_rows_of, sources, chunksize=8):
            rows += r
            skipped += s
    write_jsonl(args.out, rows)
    ext = [NZD / "step8_shared_test/build_longleg.py", SRC / "brickicl/sampling/edits.py",
           SRC / "brickicl/sampling/build_stretch.py", SRC / "brickicl/sampling/missing.py"]
    write_run_config(Path(args.out).parent, args, ext, name=f"run_config.{Path(args.out).stem}.json")
    st = [r for r in rows if r["condition"] == "stretched"]
    print(f"{len(rows)} rows ({len(st)} pairs) from {len({r['source'] for r in rows})} of {len(sources)} models "
          f"-> {args.out}")
    print("r:", dict(sorted(Counter(r["r"] for r in st).items())),
          "| k long:", dict(sorted(Counter(r["k"] for r in st).items())),
          "| reference last:", sum(r["reference_last"] for r in rows), "| jump:", sum(r["jump"] for r in rows))
    print("skipped:", dict(skipped))


if __name__ == "__main__":
    main()
