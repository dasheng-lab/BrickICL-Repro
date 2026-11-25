"""Graph edits for counterfactual contexts (README section 5). Pilot: stretch a stacked chain.

Stretching a copy (a leg, a pencil, a pillar) repeats one of its stack joints. A stack joint (X, C) is
a pair of identical parts where C sits on X through stud connections only, and everything on C's
side hangs off X through that joint alone. Inserting r new copies of X between them, each sitting on
the one below exactly as C sat on X, moves C and everything on it r stack steps further out. The
geometry stays exact: every new edge reuses the connector indices and joint parameters of an
original X-C edge, which is valid because X and C are the same part.
"""

from collections import deque

import numpy as np
from bricknet import Graph
from bricknet.graph import _components_from_edges


def _between(graph: Graph, x: int, c: int) -> np.ndarray:
    a, b = graph.edges["a"], graph.edges["b"]
    return np.flatnonzero(((a == x) & (b == c)) | ((a == c) & (b == x)))


def _reach(graph: Graph, start: int, cut: int) -> set[int]:
    """Parts reachable from start without using any edge between start and cut."""
    adj = {}
    for a, b in zip(graph.edges["a"].tolist(), graph.edges["b"].tolist()):
        if {a, b} != {start, cut}:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    seen, todo = {start}, [start]
    while todo:
        for v in adj.get(todo.pop(), []):
            if v not in seen:
                seen.add(v)
                todo.append(v)
    return seen


def stack_joint(graph: Graph, copy, anchor: int):
    """The stack joint (X, C) of a copy closest to where it attaches, or None.

    copy: node indices of the copy. anchor: the outside part it hangs from. Joints are ordered by the
    depth of X below the attachment, so isomorphic copies yield corresponding joints.
    """
    nodes = set(copy)
    depth, queue = {}, deque()
    for v in nodes:
        if len(_between(graph, anchor, v)):
            depth[v] = 0
            queue.append(v)
    adj = {}
    for a, b in zip(graph.edges["a"].tolist(), graph.edges["b"].tolist()):
        if a in nodes and b in nodes and a != b:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    while queue:
        u = queue.popleft()
        for v in sorted(adj.get(u, ())):
            if v not in depth:
                depth[v] = depth[u] + 1
                queue.append(v)
    for x in sorted(nodes, key=lambda v: (depth.get(v, 1 << 30), graph.part_ids[v], v)):
        for c in sorted(adj.get(x, ()), key=lambda v: (graph.part_ids[v], v)):
            if depth.get(c, -1) != depth[x] + 1 or graph.part_ids[c] != graph.part_ids[x]:
                continue
            rows = graph.edges[_between(graph, x, c)]
            if not (rows["family"] == "stud").all():
                continue
            if x in _reach(graph, c, x):  # C's side touches X's side elsewhere: not a clean joint
                continue
            return x, c
    return None


def tip(graph: Graph, copy, anchor: int) -> int:
    """The part of a copy farthest from where it attaches (ties: lowest part id, then index)."""
    nodes = set(copy)
    adj = {}
    for a, b in zip(graph.edges["a"].tolist(), graph.edges["b"].tolist()):
        if a in nodes and b in nodes and a != b:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    depth = {v: 0 for v in nodes if len(_between(graph, anchor, v))}
    queue = deque(sorted(depth))
    while queue:
        u = queue.popleft()
        for v in sorted(adj.get(u, ())):
            if v not in depth:
                depth[v] = depth[u] + 1
                queue.append(v)
    return min(nodes, key=lambda v: (-depth.get(v, -1), graph.part_ids[v], v))


def free_connectors(graph: Graph, part: int, kind: str, subtypes=None) -> list[int]:
    """Connector indices of `part` of the given kind ("stud" or "hole"), optionally restricted to some
    subtypes (e.g. holes that take a plain stud: "hole", "tube"), that no edge uses."""
    from bricknet.graph import _part_info

    used = set(graph.edges["a_conn"][graph.edges["a"] == part].tolist())
    used |= set(graph.edges["b_conn"][graph.edges["b"] == part].tolist())
    conns = _part_info(graph.part_ids[part]).conns
    return [k for k, c in enumerate(conns) if c[0] == kind and k not in used and (subtypes is None or c[1] in subtypes)]


def decorate(graph: Graph, sites, deco) -> tuple[Graph, list[list[int]]]:
    """Hang the same small stack of parts on every site.

    sites: (part, connector, side) per copy; side "stud" stacks the decoration on top of a free stud, side
    "hole" hangs it under a free hole. deco: [(part_id, color, hole_conn, stud_conn), ...] from the site
    outwards; each part plugs into the previous one through its hole (on top) or its stud (hung under).
    Every new edge is a straight stud joint (yaw 0). Returns the edited graph and the new node indices per site.
    """
    pids, colors = list(graph.part_ids), list(graph.colors)
    rows, added = [], []
    for part, conn, side in sites:
        prev, prev_conn, new = part, conn, []
        for pid, color, hole, stud in deco:
            node = len(pids)
            pids.append(pid)
            colors.append(color)
            row = np.zeros(1, dtype=graph.edges.dtype)[0]
            row["family"] = "stud"
            if side == "stud":  # prev's stud into node's hole; node then offers its own stud upwards
                row["a"], row["a_conn"], row["b"], row["b_conn"] = prev, prev_conn, node, hole
                prev_conn = stud
            else:  # node's stud into prev's hole; node then offers its own hole downwards
                row["a"], row["a_conn"], row["b"], row["b_conn"] = node, stud, prev, prev_conn
                prev_conn = hole
            rows.append(row)
            new.append(node)
            prev = node
        added.append(new)
    edges = np.concatenate([graph.edges, np.array(rows, dtype=graph.edges.dtype)])
    comps = _components_from_edges(len(pids), edges["a"], edges["b"])
    return Graph(tuple(pids), tuple(colors), None, edges, comps), added


def recolor(graph: Graph, nodes, color: int) -> Graph:
    """The same model with the given parts recolored. Geometry and edges are untouched."""
    nodes = set(nodes)
    colors = tuple(color if i in nodes else c for i, c in enumerate(graph.colors))
    return Graph(graph.part_ids, colors, graph.transforms, graph.edges, graph.components)


def stretch(graph: Graph, joints, r: int) -> tuple[Graph, list[list[int]]]:
    """Insert r copies of X at every joint (X, C), in the given order. New parts get indices after the
    existing ones. Returns the edited graph and the new node indices per joint."""
    pids, colors = list(graph.part_ids), list(graph.colors)
    keep = np.ones(len(graph.edges), dtype=bool)
    added_rows, added_nodes = [], []
    for x, c in joints:
        idx = _between(graph, x, c)
        keep[idx] = False
        chain = [x]
        for _ in range(r):
            chain.append(len(pids))
            pids.append(pids[x])
            colors.append(colors[x])
        added_nodes.append(chain[1:])
        chain.append(c)
        for lo, hi in zip(chain, chain[1:]):  # lo plays X's role, hi plays C's role
            for row in graph.edges[idx]:
                new = row.copy()
                new["a"] = lo if int(row["a"]) == x else hi
                new["b"] = lo if int(row["b"]) == x else hi
                added_rows.append(new)
    edges = np.concatenate([graph.edges[keep], np.array(added_rows, dtype=graph.edges.dtype)])
    comps = _components_from_edges(len(pids), edges["a"], edges["b"])
    return Graph(tuple(pids), tuple(colors), None, edges, comps), added_nodes
