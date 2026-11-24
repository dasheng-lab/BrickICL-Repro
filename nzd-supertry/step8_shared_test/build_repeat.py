"""Build the repeated-structure completion test: val models with a repeated part group, nothing edited.

Wheels on a car, lamps along a road, the legs of a table: a group of parts that hangs from the rest of the
model through one part and occurs at least twice, identically. One copy is held out as M; the other copies
stay in the input. Unlike the long-leg test (build_longleg.py) nothing is stretched, so the context and the
model's prior agree: the row only asks whether the model finds the other copy and puts the missing one in
the right place.

The rows have the format of the step 5 test set and the same text layout as the long-leg rows: a depth-first
preorder with M moved to the end and another copy last in the prefix (see build_longleg.py).

Selection: every group of `pendant_copies` with at least --min-size parts per copy, at most --groups groups
per model and --per-group held-out copies per group; models over --max-parts parts are skipped, as are
models whose walk leaves out a part of a copy.

Usage (repo root, bricknet env; run on the cluster like the other data steps):
  python nzd-supertry/step8_shared_test/build_repeat.py --out runs/nzd-supertry/eval_shared/repeat/test.jsonl
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_longleg import REPO, dataset_collisions, held_out_row, write_rows  # noqa: E402

from brickicl.sampling.build import read_graphs  # noqa: E402
from brickicl.sampling.missing import pendant_copies  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", default=str(REPO / "baseline/bricknet/data/dataset/val.npz.xz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-size", type=int, default=2, help="smallest copy, in parts")
    ap.add_argument("--max-size", type=int, default=12, help="largest copy, in parts")
    ap.add_argument("--max-parts", type=int, default=100, help="BrickNet never saw longer sequences")
    ap.add_argument("--groups", type=int, default=2, help="at most this many repeated groups per model")
    ap.add_argument("--per-group", type=int, default=2, help="at most this many held-out copies per group")
    ap.add_argument("--tries", type=int, default=20, help="walks tried per held-out copy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    graphs = read_graphs(args.graphs)
    rows, skipped, used = [], Counter(), []
    for gid, g in enumerate(graphs):
        if len(g.components) != 1:
            continue
        groups = [grp for grp in pendant_copies(g, max_size=args.max_size, min_copies=2) if grp.size >= args.min_size]
        if not groups:
            continue
        if len(g.part_ids) > args.max_parts:
            skipped[f"over {args.max_parts} parts"] += 1
            continue
        base = dataset_collisions(g, args.max_parts)
        index = 0
        for gi, grp in enumerate(groups[: args.groups]):
            copies = [list(c) for c in grp.copies]
            made = 0
            for t in range(min(len(copies), args.per_group)):
                others = [copies[j] for j in range(len(copies)) if j != t]
                row, why = held_out_row(g, copies[t], others, base, np.random.default_rng([args.seed, gid, gi, t]),
                                        args.tries, args.max_parts)
                if row is None:
                    skipped[why] += 1
                    continue
                rows.append({"source": gid, "sample_index": index, "pair": f"{gid}:{index}", "kind": "repeat",
                             "group": gi, "copies": len(copies), **row})
                index += 1
                made += 1
            if made:
                used.append((gid, len(copies), grp.size))
    write_rows(rows, used, skipped, args.out)
    ks = Counter(r["k"] for r in rows)
    print("parts per held-out copy:", dict(sorted(ks.items())))


if __name__ == "__main__":
    main()
