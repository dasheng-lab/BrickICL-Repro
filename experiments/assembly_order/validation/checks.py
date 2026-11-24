"""Step 1 checks for the masked-order augmentation (CPU only, nothing is written unless --out is given).

  walks   The shipped walks record only how many parts they placed, not which ones. Can the node set
          be recovered by replaying the path text on the graph? How long does our own collision-free
          walk take instead? Does a masked reorder of that node set stay collision-free?
  count   How many valid M does a model have? M is valid when it is connected and the rest of the
          model stays connected (at least two parts left). Exact count by size, up to --max-k.
  pool    Shape of the pt / sft walk pools, from the head of the shuffled jsonl files.

Usage (from the repo root, in the bricknet env):
  python experiments/assembly_order/validation/checks.py walks --split val --models 100
  python experiments/assembly_order/validation/checks.py count --split val
  python experiments/assembly_order/validation/checks.py pool --split pt --rows 200000
"""

import argparse
import io
import json
import lzma
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
DATA = Path(os.environ.get("BRICKNET_DATA", REPO / "baseline/bricknet/data"))
os.environ.setdefault("BRICKNET_DATA", str(DATA))

import bricknet.graph as bg  # noqa: E402
from bricknet import load_catalog, load_graphs, parse_sample, score_text, serialize_tree  # noqa: E402

from brickicl.sampling.masked_walk import masked_walk  # noqa: E402

KNOWN_COLORS = set(load_catalog().code_to_color)


def read_graphs(split):
    return load_graphs(io.BytesIO(lzma.open(DATA / "dataset" / f"{split}.npz.xz").read()))


def head_rows(split, n):
    """First n rows of a shipped walk file. Not a uniform sample: the first 200k rows of paths_pt
    cover only about 4k models, so use it for walk-level figures and the npz for model-level ones."""
    proc = subprocess.Popen(
        ["xz", "-dc", str(DATA / "dataset" / f"paths_{split}.jsonl.xz")], stdout=subprocess.PIPE, text=True
    )
    rows = []
    for line in proc.stdout:
        rows.append(json.loads(line))
        if len(rows) >= n:
            break
    proc.kill()
    return rows


def simple_adj(graph, nodes=None):
    adj = defaultdict(set)
    for a, b in zip(graph.edges["a"].tolist(), graph.edges["b"].tolist()):
        if a != b and (nodes is None or (a in nodes and b in nodes)):
            adj[a].add(b)
            adj[b].add(a)
    return adj


def connected(nodes, adj):
    nodes = set(nodes)
    if not nodes:
        return False
    start = next(iter(nodes))
    seen, stack = {start}, [start]
    while stack:
        for v in adj[stack.pop()]:
            if v in nodes and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(nodes)


# ---------------------------------------------------------------- walks


def replay(graph, tree, limit=20000):
    """Graph nodes of a parsed walk, or None. Matches each tree edge to a graph edge leaving the
    parent's node; backtracks over ties (identical twins)."""
    adj = bg._adjacency(graph.edges, set(graph.components[0]))
    steps = 0

    def extend(nodes):
        nonlocal steps
        i = len(nodes)
        if i == len(tree.parts):
            return nodes
        want, part = tree.edges[i - 1], tree.parts[i]
        u = nodes[want.parent]
        for v, ei, fwd in adj.get(u, []):
            steps += 1
            if steps > limit:
                return None
            if v in nodes or graph.part_ids[v] != part.part_id or graph.colors[v] != part.color:
                continue
            if bg._make_tree_edge(graph.edges[ei], want.parent, i, fwd, graph.part_ids) == want:
                out = extend(nodes + [v])
                if out is not None:
                    return out
        return None

    root = tree.parts[0]
    for r in graph.components[0]:
        if graph.part_ids[r] == root.part_id and graph.colors[r] == root.color:
            out = extend([r])
            if out is not None:
                return out
    return None


def own_walk(graph, seed, max_parts=100):
    """Upstream collision-free walk, returning (order, parent_of) instead of only the tree."""
    got = {}
    build = bg._build_tree

    def capture(g, order, parent_of):
        got["order"], got["parent_of"] = list(order), dict(parent_of)
        return build(g, order, parent_of)

    bg._build_tree = capture
    try:
        bg.sample_collision_free_tree(graph, seed=seed, max_parts=max_parts)
    finally:
        bg._build_tree = build
    return got["order"], got["parent_of"]


def random_subtree_M(nodes, adj, rng, k_max):
    """One valid M: a complete subtree of a random DFS spanning tree of the induced subgraph."""
    nodes = sorted(nodes)
    root = nodes[rng.integers(len(nodes))]
    parent, order, stack, seen = {}, [root], [root], {root}
    while stack:
        nbrs = [v for v in sorted(adj[stack[-1]]) if v not in seen]
        if not nbrs:
            stack.pop()
            continue
        v = nbrs[rng.integers(len(nbrs))]
        seen.add(v)
        parent[v] = stack[-1]
        order.append(v)
        stack.append(v)
    size = dict.fromkeys(order, 1)
    for v in reversed(order[1:]):
        size[parent[v]] += size[v]
    cands = [v for v in order[1:] if size[v] <= k_max and len(nodes) - size[v] >= 2]
    if not cands:
        return None
    top = cands[rng.integers(len(cands))]
    children = defaultdict(list)
    for v, p in parent.items():
        children[p].append(v)
    M, stack = [], [top]
    while stack:
        u = stack.pop()
        M.append(u)
        stack.extend(children[u])
    return M


def cmd_walks(args):
    graphs = read_graphs(args.split)
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(graphs), size=min(args.models, len(graphs)), replace=False).tolist())

    # (a) replay the shipped walks
    by_source = defaultdict(list)
    for row in head_rows(args.split, args.rows):
        if row["component_index"] == 0 and len(by_source[row["source"]]) < 2:
            by_source[row["source"]].append(row)
    ok = fail = 0
    t0 = time.time()
    for gid in pick:
        for row in by_source.get(gid, []):
            tree = parse_sample(row["path"]).tree
            nodes = replay(graphs[gid], tree)
            good = nodes is not None and len(set(nodes)) == row["nodes"]
            ok += good
            fail += not good
    print(f"(a) replay shipped walks: {ok} recovered, {fail} failed, {(time.time() - t0) / max(ok + fail, 1) * 1e3:.1f} ms each")

    # (b) our own collision-free walk, then a masked reorder of the same node set
    t_walk, t_mask, sizes, n_parts = [], [], [], []
    plain_coll = masked_coll = masked_new = n_masked = skipped = bad_color = 0
    for gid in pick:
        g = graphs[gid]
        if len(g.components) != 1:
            skipped += 1
            continue
        if any(c not in KNOWN_COLORS for c in g.colors):  # serialize_tree cannot write these (0.6% of sft)
            bad_color += 1
            continue
        t0 = time.time()
        order, parent_of = own_walk(g, seed=gid)
        t_walk.append(time.time() - t0)
        sizes.append(len(order))
        n_parts.append(len(g.part_ids))
        keep = set(order)
        if len(keep) < 4:
            continue
        plain = serialize_tree(bg._build_tree(g, order, parent_of))
        base = len(score_text(plain)[2] or [])
        plain_coll += base > 0
        adj = simple_adj(g, keep)
        for s in range(args.masks):
            M = random_subtree_M(keep, adj, np.random.default_rng(gid * 100 + s), k_max=max(1, len(keep) // 2))
            if M is None:
                continue
            t0 = time.time()
            smp = masked_walk(g, M, exclude=set(g.components[0]) - keep, seed=gid * 100 + s)
            t_mask.append(time.time() - t0)
            text = smp.prefix + smp.target
            res = parse_sample(text)
            assert res.error is None and len(res.tree.parts) == len(keep)
            c = len(score_text(text)[2] or [])
            n_masked += 1
            masked_coll += c > 0
            masked_new += c > base
    print(f"(b) own walk on {len(sizes)} models ({skipped} multi-component, {bad_color} with an uncatalogued colour skipped)")
    print(f"    walk time: median {np.median(t_walk) * 1e3:.0f} ms, p90 {np.percentile(t_walk, 90) * 1e3:.0f} ms, max {max(t_walk):.1f} s")
    print(f"    parts kept / parts in model: median {np.median(sizes):.0f} / {np.median(n_parts):.0f}; "
          f"walks that dropped parts: {sum(s < n for s, n in zip(sizes, n_parts))}")
    print(f"    plain walks with collisions under the scorer: {plain_coll}")
    print(f"    masked reorders: {n_masked}, median {np.median(t_mask) * 1e3:.1f} ms; "
          f"with any collision {masked_coll}, with more collisions than the plain walk {masked_new}")


# ---------------------------------------------------------------- count


def count_valid_M(nodes, adj, max_k, cap):
    """Exact number of valid M per size 1..max_k. Returns (counts, capped)."""
    nodes = sorted(nodes)
    n = len(nodes)
    level = {frozenset([v]) for v in nodes}
    counts, capped = {}, False
    for k in range(1, max_k + 1):
        if n - k < 2:
            break
        counts[k] = sum(connected(set(nodes) - S, adj) for S in level)
        if k == max_k:
            break
        nxt = set()
        for S in level:
            for u in S:
                for v in adj[u]:
                    if v not in S:
                        nxt.add(S | {v})
            if len(nxt) > cap:
                capped = True
                break
        if capped:
            break
        level = nxt
    return counts, capped


def cmd_count(args):
    graphs = read_graphs(args.split)
    if args.models and args.models < len(graphs):
        rng = np.random.default_rng(0)
        graphs = [graphs[i] for i in sorted(rng.choice(len(graphs), size=args.models, replace=False).tolist())]
    buckets = [(2, 5), (6, 10), (11, 20), (21, 40), (41, 100), (101, 10**9)]
    rows = defaultdict(list)
    n_capped = 0
    t0 = time.time()
    for g in graphs:
        comp = set(g.components[0])
        n = len(comp)
        adj = simple_adj(g, comp)
        counts, capped = count_valid_M(comp, adj, args.max_k, args.cap)
        n_capped += capped
        b = next(b for b in buckets if b[0] <= n <= b[1])
        half = sum(c for k, c in counts.items() if k <= n // 2)
        rows[b].append((n, counts.get(1, 0), sum(c for k, c in counts.items() if k <= 3), half, capped))
    print(f"{len(graphs)} models from {args.split}, exact for k <= {args.max_k}, {time.time() - t0:.0f} s; "
          f"{n_capped} models hit the cap of {args.cap} subsets (their counts are lower bounds)")
    print(f"{'parts':>9} {'models':>6} | valid M, k=1: median | k<=3: median | k<=min({args.max_k}, n/2): p10 / median / p90 | capped")
    for b in buckets:
        r = rows.get(b)
        if not r:
            continue
        a = np.array([x[:4] for x in r])
        label = f"{b[0]}-{b[1]}" if b[1] < 10**9 else f"{b[0]}+"
        print(f"{label:>9} {len(r):>6} | {np.median(a[:, 1]):>20.0f} | {np.median(a[:, 2]):>12.0f} | "
              f"{np.percentile(a[:, 3], 10):>10.0f} / {np.median(a[:, 3]):.0f} / {np.percentile(a[:, 3], 90):.0f} | {sum(x[4] for x in r)}")


# ---------------------------------------------------------------- pool


def cmd_pool(args):
    rows = head_rows(args.split, args.rows)
    nodes = np.array([r["nodes"] for r in rows])
    comp = np.array([r["component_nodes"] for r in rows])
    print(f"{len(rows)} walks from the head of paths_{args.split}; rounds: {dict(Counter(r['round'] for r in rows))}")
    print(f"  parts per walk: p10 {np.percentile(nodes, 10):.0f}, median {np.median(nodes):.0f}, "
          f"p90 {np.percentile(nodes, 90):.0f}, max {nodes.max()}")
    print(f"  walk covers its whole component: {np.mean([r['complete_component'] for r in rows]):.1%}; "
          f"whole model: {np.mean([r['complete_npz'] for r in rows]):.1%}")
    print(f"  walks from a component other than the largest: {np.mean([r['component_index'] > 0 for r in rows]):.1%}")
    print(f"  component larger than 100 parts: {np.mean(comp > 100):.1%} of walks")
    per_model = {}
    for r in rows:
        if r["component_index"] == 0:
            per_model[r["source"]] = r["component_nodes"]
    v = np.array(list(per_model.values()))
    print(f"  distinct models seen: {len(per_model)} (only the models at the head of the file; not representative)")
    print(f"    largest component: p10 {np.percentile(v, 10):.0f}, median {np.median(v):.0f}, "
          f"p90 {np.percentile(v, 90):.0f}; over 100 parts: {np.mean(v > 100):.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("walks")
    p.add_argument("--split", default="val")
    p.add_argument("--models", type=int, default=100)
    p.add_argument("--rows", type=int, default=50000, help="shipped walks to read for the replay test")
    p.add_argument("--masks", type=int, default=3, help="masked reorders per model")
    p.set_defaults(fn=cmd_walks)
    p = sub.add_parser("count")
    p.add_argument("--split", default="val")
    p.add_argument("--models", type=int, default=0, help="random subset (0: all)")
    p.add_argument("--max-k", type=int, default=8)
    p.add_argument("--cap", type=int, default=200000, help="stop growing a model's subsets beyond this many")
    p.set_defaults(fn=cmd_count)
    p = sub.add_parser("pool")
    p.add_argument("--split", default="pt")
    p.add_argument("--rows", type=int, default=200000)
    p.set_defaults(fn=cmd_pool)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
