"""Choosing M: one copy of a repeated pendant subassembly (pilot version of README section 4.2).

A pendant is a group of parts that hangs off the rest of the model through one part-to-part bridge
(possibly several parallel connections between the same two parts): a leg, a wheel, a lamp. Holding
out a pendant, or removing it, always leaves the rest connected. Two pendants are copies of each
other when they have the same size and the same labelled structure (part, color, connector
indices, joint parameters, and how they attach), as judged by a Weisfeiler-Lehman hash. This is a
heuristic: a VF2 check and non-pendant motifs come later with motifs.py.
"""

from collections import defaultdict
from dataclasses import dataclass

from bricknet import Graph


@dataclass(frozen=True)
class Copies:
    size: int  # parts per copy
    copies: tuple[tuple[int, ...], ...]  # disjoint node sets, sorted
    anchors: tuple[int, ...]  # the outside part each copy hangs from


def _simple_adj(graph: Graph) -> list[set[int]]:
    adj = [set() for _ in graph.part_ids]
    for a, b in zip(graph.edges["a"].tolist(), graph.edges["b"].tolist()):
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _bridges(adj: list[set[int]]) -> list[tuple[int, int]]:
    """Bridges of a simple graph (iterative Tarjan)."""
    disc, low, t, out = [-1] * len(adj), [0] * len(adj), 0, []
    for s in range(len(adj)):
        if disc[s] != -1:
            continue
        disc[s] = low[s] = t
        t += 1
        stack = [(s, -1, iter(sorted(adj[s])))]
        while stack:
            u, p, it = stack[-1]
            for v in it:
                if v == p:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = t
                    t += 1
                    stack.append((v, u, iter(sorted(adj[v]))))
                    break
                low[u] = min(low[u], disc[v])
            else:
                stack.pop()
                if p != -1:
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        out.append((p, u))
    return out


def _side(adj, start, cut, cap):
    """Nodes reachable from start without stepping onto cut; None once more than cap are found."""
    seen, stack = {start}, [start]
    while stack:
        for v in adj[stack.pop()]:
            if v != cut and v not in seen:
                seen.add(v)
                if len(seen) > cap:
                    return None
                stack.append(v)
    return seen


def _edge_label(row, x):
    """Edge label seen from endpoint x (edges are stored with canonical a/b sides)."""
    side_a = int(row["a"]) == x
    own, other = (row["a_conn"], row["b_conn"]) if side_a else (row["b_conn"], row["a_conn"])
    rot = tuple(round(float(r), 2) for r in row["rot"])
    return (str(row["family"]), side_a, int(own), int(other), bool(row["flip"]), round(float(row["yaw"]), 2), rot)


def _signature(graph: Graph, nodes: set[int], attach: int, anchor: int, iters: int = 3):
    rows = [r for r in graph.edges if int(r["a"]) in nodes or int(r["b"]) in nodes]
    inner = defaultdict(list)
    hook = []
    for r in rows:
        a, b = int(r["a"]), int(r["b"])
        if a in nodes and b in nodes:
            inner[a].append((r, b))
            inner[b].append((r, a))
        elif {a, b} == {attach, anchor}:
            hook.append(_edge_label(r, attach))
    color = {x: (graph.part_ids[x], graph.colors[x], tuple(sorted(hook)) if x == attach else ()) for x in nodes}
    for _ in range(iters):
        color = {
            x: hash((color[x], tuple(sorted((_edge_label(r, x), color[y]) for r, y in inner[x])))) for x in nodes
        }
    return len(nodes), tuple(sorted(color.values()))


def pendant_copies(graph: Graph, *, max_size: int = 8, min_copies: int = 2) -> list[Copies]:
    """Groups of identical, disjoint pendants; most copies first, then larger copies first."""
    adj = _simple_adj(graph)
    pendants = {}
    for x, y in _bridges(adj):
        for attach, anchor in ((y, x), (x, y)):
            side = _side(adj, attach, anchor, max_size)
            if side is not None and 2 * len(side) < len(adj):
                pendants[frozenset(side)] = (attach, anchor)
    groups = defaultdict(list)
    for nodes, (attach, anchor) in pendants.items():
        groups[_signature(graph, set(nodes), attach, anchor)].append((tuple(sorted(nodes)), anchor))
    out = []
    for (size, _), members in groups.items():
        used, keep = set(), []
        for nodes, anchor in sorted(members):
            if used.isdisjoint(nodes):
                used.update(nodes)
                keep.append((nodes, anchor))
        if len(keep) >= min_copies:
            out.append(Copies(size, tuple(n for n, _ in keep), tuple(a for _, a in keep)))
    return sorted(out, key=lambda c: (-len(c.copies), -c.size, c.copies))
