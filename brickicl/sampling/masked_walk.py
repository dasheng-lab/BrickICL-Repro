"""Masked frontier walk: a valid BrickNet build order with the held-out parts M placed last (README section 3).

The walk is the one BrickNet trains on (`sample_collision_free_tree`): start at a random root and
repeatedly pop a uniformly random edge from the frontier of placed parts. The only change is that
frontier edges into M are parked until every other part is placed. Every output is therefore a walk
the original sampler could also produce, just with M last. There is no collision check: the parts
keep their real poses, so the order cannot create collisions the full model does not already have.

Index rebuilding and edge parameters (including reversed edges) are left to BrickNet's own
`_build_tree` and `serialize_tree`. These are private helpers of the pinned upstream copy, so recheck
them when upstream changes.
"""

from dataclasses import dataclass

import numpy as np
from bricknet import Graph, serialize_tree
from bricknet.graph import _adjacency, _build_tree


@dataclass(frozen=True)
class Walk:
    order: tuple[int, ...]  # graph node indices in visit order (tree index = position)
    parent_of: dict  # node -> (parent node, edge index, forward), the edge the walk accepted
    n_prefix: int  # parts placed before the first part of M


@dataclass(frozen=True)
class MaskedSample:
    walk: Walk
    prefix: str  # path text of the first n_prefix parts, ends with "\n"
    target: str  # the remaining 2|M| lines (the reference completion)


def masked_walk(graph: Graph, M, *, exclude=(), seed=None, root=None, blocks=(), early=(), late=()) -> MaskedSample:
    """Walk V \\ exclude with every part of M placed after all other parts.

    M: graph node indices to hold out. exclude: nodes left out of the sequence entirely (for
    ablations). root: optional fixed root (must lie outside M and exclude). blocks: node sets to keep
    contiguous; once the walk enters a block it draws only edges into that block until the block is
    complete, so e.g. each leg appears as one run of lines (still a valid walk).
    early: context nodes drawn as soon as an edge reaches them (evidence far from M).
    late: context nodes held back until the rest of the context is placed, then placed right before M
    (evidence next to M). Raises ValueError when the request is impossible, e.g. the parts outside M
    (or outside M and late) do not form one connected piece.
    """
    M, exclude = set(map(int, M)), set(map(int, exclude))
    early, late = {int(v) for b in early for v in b}, {int(v) for b in late for v in b}
    comp = set(graph.components[0])
    if len(graph.components) != 1:
        raise ValueError("expected a single-component graph")
    if not M or not M <= comp or M & exclude:
        raise ValueError("M must be a non-empty subset of the component, disjoint from exclude")
    if (early | late) & (M | exclude) or early & late:
        raise ValueError("early and late must be disjoint context nodes")
    nodes = comp - exclude
    context = sorted(nodes - M)
    if len(context) < 2:
        raise ValueError("need at least two context parts")

    adj = _adjacency(graph.edges, nodes)
    rng = np.random.default_rng(seed)
    if root is None:
        candidates = [v for v in context if v not in late]
        root = candidates[rng.integers(len(candidates))]
    elif root not in context or root in late:
        raise ValueError("root must be a context part outside late")

    order, parent_of, placed = [root], {}, {root}
    frontier = []  # frontier edges (edge index, parent, child, forward)
    stage_of = {v: 2 for v in M} | {v: 1 for v in late}  # 0: context, 1: late context, 2: M
    stage = 0
    parked = {1: [], 2: []}  # edges into later stages, released stage by stage

    def push(u):
        for v, ei, fwd in adj.get(u, []):
            if v in nodes and v not in placed:
                s = stage_of.get(v, 0)
                (parked[s] if s > stage else frontier).append((ei, u, v, fwd))

    block_of = {v: i for i, b in enumerate(blocks) for v in b}
    left = [set(b) & nodes for b in blocks]  # unplaced nodes per block
    focus = set()  # unplaced nodes of blocks the walk has entered but not finished

    def place(child, par, ei, fwd):
        placed.add(child)
        parent_of[child] = (par, ei, fwd)
        order.append(child)
        if child in block_of:
            b = left[block_of[child]]
            b.discard(child)
            focus.update(b)
            focus.discard(child)
        push(child)

    def drain():
        while frontier:
            inside = [i for i, e in enumerate(frontier) if e[2] in focus] if focus else []
            if not inside and early:
                inside = [i for i, e in enumerate(frontier) if e[2] in early and e[2] not in placed]
            pick = inside[rng.integers(len(inside))] if inside else int(rng.integers(len(frontier)))
            ei, par, child, fwd = frontier.pop(pick)
            if child not in placed:
                place(child, par, ei, fwd)

    if root in block_of:
        left[block_of[root]].discard(root)
        focus.update(left[block_of[root]])
    push(root)
    drain()  # phase 1: the context, with edges into late nodes and M parked
    if len(order) != len(context) - len(late & nodes):
        raise ValueError("context parts are not connected without M" + (" and late" if late else ""))
    stage = 1  # phase 1b: the late context, right before M
    frontier.extend(parked[1])
    drain()
    if len(order) != len(context):
        raise ValueError("late parts are not reachable from the context")
    n_prefix = len(order)
    stage = 2  # phase 2: release M
    frontier.extend(parked[2])
    drain()
    if placed != nodes:
        raise ValueError("M is not reachable from the context")
    assert set(order[n_prefix:]) == M

    walk = Walk(tuple(order), parent_of, n_prefix)
    prefix, target = split_text(serialize_tree(_build_tree(graph, order, parent_of)), n_prefix)
    return MaskedSample(walk, prefix, target)


def split_text(text: str, n_prefix: int) -> tuple[str, str]:
    """Split path text after its first n_prefix parts (root line + 2 lines per further part)."""
    lines = text.splitlines(keepends=True)
    cut = 1 + 2 * (n_prefix - 1)
    return "".join(lines[:cut]), "".join(lines[cut:])
