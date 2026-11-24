"""Masked-order augmentation: paired sequences of the same tree, with one subtree moved to the end.

BrickNet's shipped training walks are depth-first preorders of a spanning tree: every part attaches
to the previous part or to one of its ancestors (step4_train/README.md). In a preorder the subtree
of any part is one contiguous block. For one model the sampler draws several pairs:

  plain   the tree of one walk of BrickNet's own sampler (`sample_collision_free_tree`: at most
          100 parts, collision-free), written in preorder with siblings in random order. This is
          the order the frozen model was trained on.
  masked  the same tree, the same sibling order, but the block of one subtree M is moved to the
          end:  A + M + B  ->  A + B + M.

So the two arms share parts, edges and geometry; they differ only in where M is written. In the
masked order the first part of M usually attaches to a part that is off the current root-to-last
path, which never happens in a preorder (`jump`). When M hangs from the last context part or one
of its ancestors the masked order is itself a preorder; those rows are kept and flagged
`jump: false`. Both arms are ordinary path text for PT-style training (loss on the whole sequence,
no marker where M starts).

M is a complete subtree of the walk's tree, so M and the rest are both connected by construction.
Because nothing but the order changes, both arms keep the walk's collision-free geometry.
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("BRICKNET_DATA", str(REPO / "baseline/bricknet/data"))  # collider meshes live in cvx/

import bricknet.graph as bg  # noqa: E402
from bricknet import load_catalog, parse_sample, score_text, serialize_tree  # noqa: E402

_KNOWN_COLORS = None


def serializable(graph) -> bool:
    """False when a part has a colour the catalog cannot name (0.6% of sft models)."""
    global _KNOWN_COLORS
    if _KNOWN_COLORS is None:
        _KNOWN_COLORS = set(load_catalog().code_to_color)
    return all(c in _KNOWN_COLORS for c in graph.colors)


def collision_free_walk(graph, seed, max_parts=100):
    """BrickNet's own walk, returning (order, parent_of, path text) in the sampler's visit order.

    Upstream returns only the tree; the node order is taken from its call to `_build_tree`, so the
    walk is exactly the upstream one. parent_of: node -> (parent node, edge index, forward).
    Not thread-safe (use processes).
    """
    got = {}
    build = bg._build_tree

    def capture(g, order, parent_of):
        got["order"], got["parent_of"] = list(order), dict(parent_of)
        return build(g, order, parent_of)

    bg._build_tree = capture
    try:
        tree = bg.sample_collision_free_tree(graph, seed=seed, max_parts=max_parts)
    finally:
        bg._build_tree = build
    return got["order"], got["parent_of"], serialize_tree(tree)


def simple_adj(graph, nodes):
    """Neighbour sets of the simple graph induced on `nodes` (parallel edges merged)."""
    adj = defaultdict(set)
    for a, b in zip(graph.edges["a"].tolist(), graph.edges["b"].tolist()):
        if a != b and a in nodes and b in nodes:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def is_connected(nodes, adj) -> bool:
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


def preorder(order, parent_of, rng):
    """Depth-first preorder of the walk's tree, siblings in random order (as in the shipped walks)."""
    kids = defaultdict(list)
    for v in order[1:]:
        kids[parent_of[v][0]].append(v)
    out, stack = [], [order[0]]
    while stack:
        u = stack.pop()
        out.append(u)
        ks = kids[u]
        stack.extend(ks[i] for i in rng.permutation(len(ks)))
    return out


def subtree_sizes(pre, parent_of):
    """Parts in the subtree of each node. In a preorder, the subtree of pre[i] is pre[i : i + size]."""
    size = dict.fromkeys(pre, 1)
    for v in reversed(pre[1:]):
        size[parent_of[v][0]] += size[v]
    return size


def choose_block(pre, size, rng, k_max):
    """Start index of a subtree block to hold out, or None. The size aims at a uniform draw from
    1..k_max. A block that is already last is not a candidate: moving it would change nothing."""
    n = len(pre)
    cands = [i for i in range(1, n) if size[pre[i]] <= k_max and i + size[pre[i]] < n]
    if not cands:
        return None
    want = int(rng.integers(1, k_max + 1))
    best = min(abs(size[pre[i]] - want) for i in cands)
    tops = [i for i in cands if abs(size[pre[i]] - want) == best]
    return tops[rng.integers(len(tops))]


def n_pairs_for(n_parts, parts_per_M=4, max_M=25):
    """How many M to draw for a model: about one per `parts_per_M` parts (step1_checks/README.md, check 2)."""
    return int(min(max(round(n_parts / parts_per_M), 1), max_M))


def sample_pairs(graph, n_pairs, seed, *, max_parts=100, max_frac=0.5, min_parts=6, retries=4):
    """Up to n_pairs (plain, masked) pairs for one single-component model.

    Returns (pairs, stats). A pair is {"plain": {...}, "masked": {...}}; both rows carry `path`.
    Deterministic in (graph, seed). A pair is skipped when its plain text, its masked text or its
    (parts, M) repeats an earlier pair of this model, so small models yield fewer pairs than asked.
    """
    if len(graph.components) != 1:
        raise ValueError("expected a single-component graph")
    stats, pairs = Counter(), []
    seen_plain, seen_masked, seen_M = set(), set(), set()
    for j in range(n_pairs):
        rng = np.random.default_rng([int(seed), j])
        for _ in range(retries):
            order, parent_of, _ = collision_free_walk(graph, int(rng.integers(2**31)), max_parts)
            n = len(order)
            if n < min_parts:
                why = "walk_too_small"
                continue
            pre = preorder(order, parent_of, rng)
            plain = serialize_tree(bg._build_tree(graph, pre, parent_of))
            if plain in seen_plain:
                why = "plain_repeat"
                continue
            size = subtree_sizes(pre, parent_of)
            i = choose_block(pre, size, rng, max(1, int(n * max_frac)))
            if i is None:
                why = "no_block"  # e.g. the tree is a single chain
                continue
            k = size[pre[i]]
            M = pre[i : i + k]
            moved = pre[:i] + pre[i + k :] + M
            masked = serialize_tree(bg._build_tree(graph, moved, parent_of))
            if masked in seen_masked or (frozenset(pre), frozenset(M)) in seen_M:
                why = "masked_repeat"
                continue
            for text in (plain, masked):
                res = parse_sample(text)
                if res.error is not None or len(res.tree.parts) != n:
                    raise AssertionError(f"sequence does not parse back: {res.error}")
            # same tree as the collision-checked walk, so this should never fire ([] clean, None: a
            # collider mesh is missing, kept as upstream does)
            if score_text(plain)[2] or score_text(masked)[2]:
                why = "collides"
                continue
            break
        else:
            stats[why] += 1
            continue

        seen_plain.add(plain)
        seen_masked.add(masked)
        seen_M.add((frozenset(pre), frozenset(M)))
        anchor = parent_of[M[0]][0]  # the context part that M hangs from
        # If the anchor is the last context part or one of its ancestors, the masked order is still a
        # preorder (M is just visited after its later siblings) and holds no off-path attachment.
        u, on_path = moved[n - k - 1], False
        while True:
            if u == anchor:
                on_path = True
                break
            if u not in parent_of:
                break
            u = parent_of[u][0]
        stats["jump" if not on_path else "no_jump"] += 1
        pairs.append(
            {
                "plain": {
                    "nodes": n,
                    "order": pre,
                    "block_start": i,
                    "k": k,
                    "lag": i - pre.index(anchor),
                    "path": plain,
                },
                "masked": {
                    "nodes": n,
                    "order": moved,
                    "M": M,
                    "k": k,
                    "n_prefix": n - k,
                    "lag": (n - k) - moved.index(anchor),
                    "jump": not on_path,
                    "path": masked,
                },
            }
        )
        stats["pairs"] += 1
    return pairs, stats
