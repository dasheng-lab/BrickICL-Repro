"""Build the stretch probe: does the model complete a held-out leg as long (context) or short (prior)?

For each val model with a repeated, stretchable pendant (legs, pencils, rockets ...), one copy is held
out and every row offers two completions of the same prefix:
  target_short  the copy as built
  target_long   the copy stretched by r stack steps
Conditions differ only in what the prefix shows of the other copies:
  original   unchanged                   -> the prior and the context agree on short
  stretched  stretched by r              -> the context says long, the prior says short
  ablated    removed                     -> no context, prior only
Models whose stretched variant collides (beyond collisions already in the original) are skipped.

Usage:
  python -m brickicl.sampling.build_stretch --graphs baseline/bricknet/data/dataset/val.npz.xz --out stretch.jsonl
"""

import argparse
import json

from bricknet import score_text

from .build import read_graphs
from .edits import stack_joint, stretch
from .masked_walk import masked_walk
from .missing import pendant_copies


def stretchable_group(graph, max_size, min_copies):
    """First pendant group whose copies all have a stack joint, with those joints."""
    for grp in pendant_copies(graph, max_size=max_size, min_copies=min_copies):
        joints = [stack_joint(graph, c, a) for c, a in zip(grp.copies, grp.anchors)]
        if all(joints):
            return grp, joints
    return None, None


def n_collisions(text):
    return len(score_text(text)[2] or [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=2, help="stack steps added per stretched copy")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-copies", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=12)
    ap.add_argument("--layout", choices=["scattered", "contiguous"], default="scattered",
                    help="scattered: plain BrickNet walk (copies interleave with other parts); "
                         "contiguous: each other copy is written as one run of lines")
    ap.add_argument("--max-parts", type=int, default=100, help="BrickNet never saw longer sequences")
    args = ap.parse_args()

    graphs = read_graphs(args.graphs)
    n_rows, used, skipped = 0, [], {}
    with open(args.out, "w") as f:
        for gid, g in enumerate(graphs):
            grp, joints = stretchable_group(g, args.max_size, args.min_copies)
            if grp is None:
                continue
            if len(g.part_ids) + args.r * len(grp.copies) > args.max_parts:
                skipped[gid] = f"over {args.max_parts} parts once stretched"
                continue
            first = masked_walk(g, grp.copies[0], seed=0)
            base = n_collisions(first.prefix + first.target)  # collisions already in the original model
            ok = True
            rows = []
            for t, copy in enumerate(grp.copies):
                others = [j for j in range(len(grp.copies)) if j != t]
                other_nodes = sorted(v for j in others for v in grp.copies[j])
                g_t, (new_t,) = stretch(g, [joints[t]], args.r)  # only the held-out copy stretched
                g_o, _ = stretch(g, [joints[j] for j in others], args.r)  # the others stretched
                g_all, added = stretch(g, [joints[j] for j in others] + [joints[t]], args.r)
                short_M, long_t, long_all = list(copy), list(copy) + new_t, list(copy) + added[-1]
                contiguous = args.layout == "contiguous"
                plain = [grp.copies[j] for j in others] if contiguous else []
                grown = [list(grp.copies[j]) + added[i] for i, j in enumerate(others)] if contiguous else []
                variants = {  # condition -> (graph for short, graph for long, long M, excluded, blocks)
                    "original": (g, g_t, long_t, (), plain),
                    "stretched": (g_o, g_all, long_all, (), grown),
                    "ablated": (g, g_t, long_t, other_nodes, []),
                }
                for s in range(args.seeds):
                    seed = gid * 1000 + t * 10 + s
                    for cond, (gs, gl, long_M, excl, blocks) in variants.items():
                        sh = masked_walk(gs, short_M, exclude=excl, seed=seed, blocks=blocks)
                        lo = masked_walk(gl, long_M, exclude=excl, seed=seed, blocks=blocks)
                        assert sh.prefix == lo.prefix, (gid, t, cond)
                        if s == 0 and cond == "stretched" and n_collisions(lo.prefix + lo.target) > base:
                            ok = False  # the fully stretched model is not physically valid
                        rows.append({
                            "source": gid,
                            "pair": f"{gid}-{t}-{s}",
                            "condition": cond,
                            "layout": args.layout,
                            "r": args.r,
                            "copies": len(grp.copies),
                            "n_prefix": sh.walk.n_prefix,
                            "k_short": len(short_M),
                            "k_long": len(long_M),
                            "prefix": sh.prefix,
                            "target_short": sh.target,
                            "target_long": lo.target,
                            "seed": seed,
                        })
            if not ok:
                skipped[gid] = "stretched model collides"
                continue
            used.append((gid, len(grp.copies), grp.size))
            for row in rows:
                f.write(json.dumps(row) + "\n")
            n_rows += len(rows)
    print(f"{n_rows} rows from {len(used)} models -> {args.out}")
    print("models (id, copies, parts per copy):", used)
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
