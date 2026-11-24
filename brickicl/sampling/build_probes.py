"""Two-answer context probes beyond the long-leg test.

Every row holds one prefix and two completions of the held-out copy:
  target_prior  what the training prior suggests (the copy as built)
  target_ctx    what the edited context suggests
Rows sharing `pair` (same model, held-out copy and walk seed) differ only in `variant`, i.e. in what the
prefix shows of the other copies; `ref_variant` names the unedited one. Other copies are always written
contiguously (the layout under which frozen BrickNet followed stretched legs).

Probes:
  color     the other copies recolored to a colour the model does not contain
            variants: original / recolored / ablated (other copies removed)
  count     models with >= 3 copies: k of the other copies stretched by r (k = 0 .. all), in a fixed
            random order, so k = 2 of 3 is "2 stretched + 1 original" (majority)
            variants: s0 (reference) .. s<k>
  position  the other copies placed first in the prefix (early) or right before the held-out copy (late)
            groups early / late, variants: original / stretched by r
  novel     every other copy carries the same small decoration (a 1x1 round plate and a 1x1 cone in
            colours the model does not contain) on a free connector of its tip
            variants: original / decorated / ablated
  where     pilot-style rows (one target, for brickicl.eval.frozen): the anchor the held-out copy hangs
            from is placed at a random position, first in the prefix, or last (right before the copy)
            conditions: anchor_random / anchor_first / anchor_last

Usage:
  python -m brickicl.sampling.build_probes --probe color --graphs .../val.npz.xz --out probes.color.jsonl
"""

import argparse
import json

import numpy as np
from bricknet import parse_sample
from bricknet.data import load_catalog

from .build import read_graphs
from .build_stretch import n_collisions, stretchable_group
from .edits import decorate, free_connectors, recolor, stretch, tip
from .masked_walk import masked_walk
from .missing import pendant_copies

# vivid colours that are rare in the dataset, tried in this order (first one absent from the model wins)
PALETTE = ["lime", "magenta", "medium azure", "orange", "bright green", "red", "blue", "yellow"]
# decoration for the novel probe: (part name, hole connector, stud connector), from the copy outwards
DECO = [("plate 1x1 round", 0, 1), ("cone 1x1", 3, 4)]


def pair_rows(gid, t, s, variants, short_M, long_M, extra):
    """Walk each variant twice (prior and ctx answer) and return one row per variant.
    variants: name -> (graph_prior, graph_ctx, kwargs for masked_walk)."""
    rows = []
    for name, (gp, gc, kw) in variants.items():
        p = masked_walk(gp, short_M, seed=gid * 1000 + t * 10 + s, **kw)
        c = masked_walk(gc, long_M, seed=gid * 1000 + t * 10 + s, **kw)
        assert p.prefix == c.prefix, (gid, t, s, name)
        rows.append({
            "source": gid,
            "pair": f"{gid}-{t}-{s}",
            "variant": name,
            "n_prefix": p.walk.n_prefix,
            "k_prior": len(short_M),
            "k_ctx": len(long_M),
            "prefix": p.prefix,
            "target_prior": p.target,
            "target_ctx": c.target,
        } | extra)
    return rows


def color_rows(graphs, args):
    cat = load_catalog()
    code = {name: c for c, name in cat.code_to_color.items()}
    for gid, g in enumerate(graphs):
        if any(c not in cat.code_to_color for c in g.colors):
            continue  # a few val parts carry colour codes outside the vocabulary
        groups = [c for c in pendant_copies(g, max_size=args.max_size, min_copies=2) if c.size >= args.min_size]
        if not groups:
            continue
        grp = groups[0]
        present = set(g.colors)
        color = next(code[n] for n in PALETTE if code[n] not in present)
        for t, copy in enumerate(grp.copies):
            others = [v for j, c in enumerate(grp.copies) if j != t for v in c]
            blocks = [c for j, c in enumerate(grp.copies) if j != t]
            g_t, g_o, g_all = recolor(g, copy, color), recolor(g, others, color), recolor(g, list(copy) + others, color)
            extra = {"probe": "color", "group": "", "ref_variant": "original", "copies": len(grp.copies),
                     "color": cat.code_to_color[color]}
            for s in range(args.seeds):
                yield from pair_rows(gid, t, s, {
                    "original": (g, g_t, {"blocks": blocks}),
                    "recolored": (g_o, g_all, {"blocks": blocks}),
                    "ablated": (g, g_t, {"exclude": others}),
                }, copy, copy, extra)


def stretched_models(graphs, args, min_copies):
    """Stretchable groups that stay within 100 parts and collision-free when every copy is stretched."""
    for gid, g in enumerate(graphs):
        grp, joints = stretchable_group(g, args.max_size, min_copies)
        if grp is None or len(g.part_ids) + args.r * len(grp.copies) > 100:
            continue
        full, _ = stretch(g, joints, args.r)
        first = masked_walk(g, grp.copies[0], seed=0)
        whole = masked_walk(full, list(grp.copies[0]) + [len(g.part_ids) + i for i in range(args.r)], seed=0)
        if n_collisions(whole.prefix + whole.target) > n_collisions(first.prefix + first.target):
            continue
        yield gid, g, grp, joints


def count_rows(graphs, args):
    for gid, g, grp, joints in stretched_models(graphs, args, min_copies=3):
        for t, copy in enumerate(grp.copies):
            others = [j for j in range(len(grp.copies)) if j != t]
            order = [others[i] for i in np.random.default_rng(gid * 100 + t).permutation(len(others))]
            extra = {"probe": "count", "group": "", "ref_variant": "s0", "copies": len(grp.copies), "r": args.r}
            variants = {}
            for k in range(len(others) + 1):
                chosen = order[:k]
                g_k, added = stretch(g, [joints[j] for j in chosen], args.r)
                g_kl, added_l = stretch(g, [joints[j] for j in chosen] + [joints[t]], args.r)
                grown = {j: list(grp.copies[j]) + added[chosen.index(j)] if j in chosen else list(grp.copies[j])
                         for j in others}
                variants[f"s{k}"] = (g_k, g_kl, {"blocks": list(grown.values())}, list(copy) + added_l[-1])
            for s in range(args.seeds):
                for name, (gp, gc, kw, long_M) in variants.items():
                    yield from pair_rows(gid, t, s, {name: (gp, gc, kw)}, copy, long_M, extra | {"k_stretched": int(name[1:])})


def position_rows(graphs, args):
    for gid, g, grp, joints in stretched_models(graphs, args, min_copies=2):
        for t, copy in enumerate(grp.copies):
            others = [j for j in range(len(grp.copies)) if j != t]
            g_t, (new_t,) = stretch(g, [joints[t]], args.r)
            g_o, _ = stretch(g, [joints[j] for j in others], args.r)
            g_all, added = stretch(g, [joints[j] for j in others] + [joints[t]], args.r)
            plain = [grp.copies[j] for j in others]
            grown = [list(grp.copies[j]) + added[i] for i, j in enumerate(others)]
            for where in ("early", "late"):
                extra = {"probe": "position", "group": where, "ref_variant": "original", "copies": len(grp.copies),
                         "r": args.r}
                for s in range(args.seeds):
                    rows = pair_rows(gid, t, s, {"original": (g, g_t, {"blocks": plain, where: plain})}, copy,
                                     list(copy) + new_t, extra)
                    rows += pair_rows(gid, t, s, {"stretched": (g_o, g_all, {"blocks": grown, where: grown})}, copy,
                                      list(copy) + added[-1], extra)
                    yield from rows


def novel_rows(graphs, args):
    cat = load_catalog()
    code = {name: c for c, name in cat.code_to_color.items()}
    pid = {}
    for i, name in enumerate(cat.id_to_name):
        pid.setdefault(name, i)
    for gid, g in enumerate(graphs):
        if any(c not in cat.code_to_color for c in g.colors):
            continue
        groups = [c for c in pendant_copies(g, max_size=args.max_size, min_copies=2) if c.size >= args.min_size]
        if not groups:
            continue
        grp = groups[0]
        tips = [tip(g, c, a) for c, a in zip(grp.copies, grp.anchors)]
        if len({g.part_ids[x] for x in tips}) != 1 or len(g.part_ids) + len(DECO) * len(tips) > 100:
            continue
        sites = None
        for side, mates in (("stud", ("stud", "open")), ("hole", ("hole", "tube"))):  # same free connector on every tip
            shared = set.intersection(*(set(free_connectors(g, x, side, mates)) for x in tips))
            if shared:
                sites = [(x, min(shared), side) for x in tips]
                break
        if sites is None:
            continue
        absent = [code[n] for n in PALETTE if code[n] not in set(g.colors)]
        deco = [(pid[name], absent[i], hole, stud) for i, (name, hole, stud) in enumerate(DECO)]
        full, added_all = decorate(g, sites, deco)
        base_walk = masked_walk(g, grp.copies[0], seed=0)
        full_walk = masked_walk(full, list(grp.copies[0]) + added_all[0], seed=0)
        if parse_sample(full_walk.prefix + full_walk.target).error is not None:
            continue  # BrickNet cannot express one of the new joints
        if n_collisions(full_walk.prefix + full_walk.target) > n_collisions(base_walk.prefix + base_walk.target):
            continue  # the decoration overlaps the model somewhere
        for t, copy in enumerate(grp.copies):
            others = [j for j in range(len(grp.copies)) if j != t]
            other_nodes = [v for j in others for v in grp.copies[j]]
            g_t, (new_t,) = decorate(g, [sites[t]], deco)
            g_o, _ = decorate(g, [sites[j] for j in others], deco)
            g_all, added = decorate(g, [sites[j] for j in others] + [sites[t]], deco)
            plain = [grp.copies[j] for j in others]
            grown = [list(grp.copies[j]) + added[i] for i, j in enumerate(others)]
            extra = {"probe": "novel", "group": "", "ref_variant": "original", "copies": len(grp.copies),
                     "decoration": [f"{name} | {cat.code_to_color[c]}" for (_, c, _, _), (name, _, _) in zip(deco, DECO)]}
            for s in range(args.seeds):
                rows = pair_rows(gid, t, s, {"original": (g, g_t, {"blocks": plain})}, copy, list(copy) + new_t, extra)
                rows += pair_rows(gid, t, s, {"decorated": (g_o, g_all, {"blocks": grown})}, copy,
                                  list(copy) + added[-1], extra)
                rows += pair_rows(gid, t, s, {"ablated": (g, g_t, {"exclude": other_nodes})}, copy,
                                  list(copy) + new_t, extra)
                yield from rows


def where_rows(graphs, args):
    """Pilot-style rows: one target, the anchor of the held-out copy first / last / anywhere in the prefix."""
    for gid, g in enumerate(graphs):
        groups = [c for c in pendant_copies(g, max_size=8, min_copies=3) if c.size >= 2]
        if not groups:
            continue
        grp = groups[0]
        for t, copy in enumerate(grp.copies):
            anchor = grp.anchors[t]
            blocks = [c for j, c in enumerate(grp.copies) if j != t]
            for s in range(args.seeds):
                seed = gid * 1000 + t * 10 + s
                try:
                    walks = {
                        "anchor_random": masked_walk(g, copy, seed=seed, blocks=blocks),
                        "anchor_first": masked_walk(g, copy, seed=seed, blocks=blocks, root=anchor),
                        "anchor_last": masked_walk(g, copy, seed=seed, blocks=blocks, late=[[anchor]]),
                    }
                except ValueError:
                    continue  # the anchor holds the rest together, so it cannot come last
                for cond, w in walks.items():
                    pos = w.walk.order.index(anchor)
                    yield {
                        "source": gid, "pair": f"{gid}-{t}-{s}", "strategy": "where", "condition": cond,
                        "copies": len(grp.copies), "M": list(copy), "n_prefix": w.walk.n_prefix, "k": len(copy),
                        "anchor_pos": pos / max(w.walk.n_prefix - 1, 1), "anchor_lag": w.walk.n_prefix - 1 - pos,
                        "prefix": w.prefix, "target": w.target, "seed": seed,
                    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", choices=["color", "count", "position", "novel", "where"], required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=4, help="stack steps for count / position")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=2, help="parts per copy (color)")
    ap.add_argument("--max-size", type=int, default=12)
    args = ap.parse_args()

    graphs = read_graphs(args.graphs)
    make = {"color": color_rows, "count": count_rows, "position": position_rows, "novel": novel_rows,
            "where": where_rows}[args.probe]
    n, models = 0, set()
    with open(args.out, "w") as f:
        for row in make(graphs, args):
            f.write(json.dumps(row) + "\n")
            n += 1
            models.add(row["source"])
    print(f"{args.probe}: {n} rows from {len(models)} models -> {args.out}")


if __name__ == "__main__":
    main()
