"""Build the pilot ICL set: hold out one copy of a repeated pendant (a wheel, a leg) and pair two contexts.

  evidence  the other copies stay in the prefix (the model can copy them)
  ablated   the other copies are removed from the model entirely (only the prior is left)

Both rows of a pair share the held-out copy M and the walk seed, and the target is always the original M.

Usage:
  python -m brickicl.sampling.build --graphs data/bricknet/dataset/val.npz.xz --out pilot.jsonl
"""

import argparse
import io
import json
import lzma
from collections import Counter

from bricknet import load_graphs, parse_sample

from .masked_walk import masked_walk
from .missing import pendant_copies


def read_graphs(path):
    data = lzma.open(path).read() if str(path).endswith(".xz") else open(path, "rb").read()
    return load_graphs(io.BytesIO(data))


def check(sample, n_parts):
    """The full text must parse back to exactly the walked parts."""
    res = parse_sample(sample.prefix + sample.target)
    assert res.error is None, res.error
    assert len(res.tree.parts) == n_parts
    assert sample.prefix.endswith("\n") and sample.target.count("\n") == 2 * (n_parts - sample.walk.n_prefix)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", required=True, help="graphs .npz or .npz.xz (e.g. val.npz.xz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-copies", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=2, help="parts per copy")
    ap.add_argument("--max-size", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3, help="walk orders per held-out copy")
    args = ap.parse_args()

    graphs = read_graphs(args.graphs)
    n_rows, kinds = 0, Counter()
    with open(args.out, "w") as f:
        for gid, g in enumerate(graphs):
            groups = [c for c in pendant_copies(g, max_size=args.max_size, min_copies=args.min_copies)]
            groups = [c for c in groups if c.size >= args.min_size]
            if not groups:
                continue
            grp = groups[0]  # one group per model: most copies, then largest copies
            kinds[(len(grp.copies), grp.size)] += 1
            for t, M in enumerate(grp.copies):
                others = sorted(i for j, c in enumerate(grp.copies) if j != t for i in c)
                for s in range(args.seeds):
                    seed = gid * 1000 + t * 10 + s
                    for condition, exclude in (("evidence", ()), ("ablated", others)):
                        smp = masked_walk(g, M, exclude=exclude, seed=seed)
                        check(smp, len(smp.walk.order))
                        row = {
                            "source": gid,
                            "pair": f"{gid}-{t}-{s}",
                            "strategy": "pendant-copy",
                            "condition": condition,
                            "copies": len(grp.copies),
                            "M": list(M),
                            "exclude": list(exclude),
                            "order": list(smp.walk.order),
                            "n_prefix": smp.walk.n_prefix,
                            "k": len(M),
                            "prefix": smp.prefix,
                            "target": smp.target,
                            "seed": seed,
                        }
                        f.write(json.dumps(row) + "\n")
                        n_rows += 1
    print(f"{n_rows} rows from {sum(kinds.values())} models -> {args.out}")
    print("(copies, parts per copy): models =", dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
