"""Build the long-leg completion test: val models whose repeated limb has been stretched.

For each val model with a repeated, stretchable pendant (legs, pencils, rockets ...; the same selection as
brickicl/sampling/build_stretch.py), every copy is made r stacked parts longer. The model then looks unlike
anything in the training data: BrickNet's prior says the limb is short, the context shows long ones. One copy
is held out as M, and the reference answer is the long copy, which is what the context calls for.

A model is skipped when stretching makes parts collide. BrickNet's walk leaves out parts that collide; a walk
is accepted only if it leaves out no part of a limb and no more parts than a walk of the unedited model does
(val #114, the fox, has one such part in the dataset itself).

The rows have the format of the step 5 test set (prefix = G \\ M, target = M), so every model is run on them
with completion_evaluation/eval_completion.py. The text is written so that one and the same prefix suits all four models:

  - it is a depth-first preorder of a BrickNet walk of the edited model, with M moved to the end. That is the
    order the frozen model, the control and the masked-order model were trained on (sampling);
  - among siblings, the branch that leads to another copy of the limb is visited last, so that copy is the
    last thing in the prefix, right before M. That is the "reference last" layout of the teammate's model.

Usage (repo root, bricknet env; run on the cluster like the other data steps):
  python training/assembly_order/robustness_evaluation/build_longleg.py --out runs/assembly_order/eval_shared/longleg/test.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "sampling"))

import bricknet.graph as bg  # noqa: E402
from bricknet import parse_sample, score_text, serialize_tree  # noqa: E402
from sampler import collision_free_walk, subtree_sizes  # noqa: E402

from brickicl.sampling.build import read_graphs  # noqa: E402
from brickicl.sampling.build_stretch import stretchable_group  # noqa: E402
from brickicl.sampling.edits import stretch  # noqa: E402


def order_with_reference_last(order, parent_of, M, reference, rng):
    """Preorder of the walk's tree without M's subtree, the branch towards `reference` visited last at every
    level, then M's subtree in preorder. Returns None when M is not one whole subtree of this tree."""
    kids = defaultdict(list)
    for v in order[1:]:
        kids[parent_of[v][0]].append(v)
    roots = [v for v in M if v == order[0] or parent_of[v][0] not in M]
    if len(roots) != 1 or roots[0] == order[0]:
        return None
    top = roots[0]
    late = set()                       # the reference copy and its ancestors
    for v in reference:
        while v not in late:
            late.add(v)
            if v == order[0]:
                break
            v = parent_of[v][0]

    def walk(start, skip):
        out, stack = [], [start]
        while stack:
            u = stack.pop()
            out.append(u)
            ks = [k for k in kids[u] if k != skip]
            ks = [ks[i] for i in rng.permutation(len(ks))]
            stack.extend(sorted(ks, key=lambda k: k not in late))   # popped from the end: late branch last
        return out

    body, block = walk(order[0], top), walk(top, None)
    return body + block if set(block) == set(M) else None


def held_out_row(graph, M, others, base, rng, tries, max_parts):
    """Write `graph` as "everything but M, then M" with another copy last in the prefix.

    M: the parts of the held-out copy; others: the other copies; base: how many parts BrickNet's walk leaves
    out of the unedited model. Returns (fields of a test row, None) or (None, why it could not be built).
    """
    in_copy = set(M).union(*map(set, others))
    why = "no walk keeps M as one subtree"
    for _ in range(tries):
        order, parent_of, _ = collision_free_walk(graph, int(rng.integers(2**31)), max_parts)
        total, dropped = len(order), set(range(len(graph.part_ids))) - set(order)
        if len(dropped) > base or dropped & in_copy:
            why = "the model collides or is too large (a walk drops parts)"
            continue
        reference = others[int(rng.integers(len(others)))]
        moved = order_with_reference_last(order, parent_of, set(M), reference, rng)
        if moved is None:
            continue
        text = serialize_tree(bg._build_tree(graph, moved, parent_of))
        res = parse_sample(text)
        assert res.error is None and len(res.tree.parts) == total, res.error
        if score_text(text)[2]:
            why = "the model collides"
            continue
        break
    else:
        return None, why
    k, n = len(M), total
    lines = text.splitlines(keepends=True)
    a = 1 + 2 * (n - k - 1)
    assert len(lines) == a + 2 * k and set(moved[n - k:]) == set(M)
    anchor = parent_of[moved[n - k]][0]
    u, on_path = moved[n - k - 1], False                   # is M's anchor on the current path?
    while True:
        if u == anchor:
            on_path = True
            break
        if u not in parent_of:
            break
        u = parent_of[u][0]
    assert subtree_sizes(moved, parent_of)[moved[n - k]] == k
    return {"k": k, "nodes": n, "n_prefix": n - k, "left_out": len(graph.part_ids) - n,
            "reference_last": moved[n - k - 1] in set(reference), "jump": not on_path,
            "lag": (n - k) - moved.index(anchor),
            "prefix": "".join(lines[:a]), "target": "".join(lines[a:])}, None


def dataset_collisions(graph, max_parts):
    """Parts BrickNet's walk leaves out of an unedited model because they collide in the dataset itself."""
    return min(len(graph.part_ids) - len(collision_free_walk(graph, s, max_parts)[0]) for s in range(3))


def write_rows(rows, used, skipped, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows from {len(used)} val models -> {out}")
    print("models (val id, copies, parts per copy):", used)
    print("reference copy is last in the prefix:", sum(r["reference_last"] for r in rows), "of", len(rows))
    print("off-path attachment:", sum(r["jump"] for r in rows), "of", len(rows))
    print("skipped:", dict(skipped))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", default=str(REPO / "baseline/bricknet/data/dataset/val.npz.xz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=4, help="stacked parts added to every copy")
    ap.add_argument("--max-size", type=int, default=12, help="largest copy, in parts, before stretching")
    ap.add_argument("--max-parts", type=int, default=100, help="BrickNet never saw longer sequences")
    ap.add_argument("--per-model", type=int, default=2, help="at most this many held-out copies per model")
    ap.add_argument("--tries", type=int, default=20, help="walks tried per held-out copy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    graphs = read_graphs(args.graphs)
    rows, skipped, used = [], Counter(), []
    for gid, g in enumerate(graphs):
        grp, joints = stretchable_group(g, args.max_size, 2)
        if grp is None:
            continue
        n_copies = len(grp.copies)
        if len(g.part_ids) + args.r * n_copies > args.max_parts:
            skipped[f"over {args.max_parts} parts once stretched"] += 1
            continue
        g_long, added = stretch(g, joints, args.r)                 # every copy stretched
        limbs = [list(c) + new for c, new in zip(grp.copies, added)]
        base = dataset_collisions(g, args.max_parts)
        made = 0
        for t in range(min(n_copies, args.per_model)):
            others = [limbs[j] for j in range(n_copies) if j != t]
            row, why = held_out_row(g_long, limbs[t], others, base, np.random.default_rng([args.seed, gid, t]),
                                    args.tries, args.max_parts)
            if row is None:
                skipped[why] += 1
                continue
            rows.append({"source": gid, "sample_index": t, "pair": f"{gid}:{t}", "kind": "longleg", "r": args.r,
                         "copies": n_copies, "k_short": len(grp.copies[t]), **row})
            made += 1
        if made:
            used.append((gid, n_copies, grp.size))
    write_rows(rows, used, skipped, args.out)


if __name__ == "__main__":
    main()
