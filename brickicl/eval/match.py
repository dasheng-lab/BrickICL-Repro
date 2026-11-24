"""Order-invariant scoring of a completion against the held-out parts (sampling README section 7).

The prefix is shared, so the reference (prefix + target) and the completion (prefix + generated)
realize in the same frame: `decode_graph` roots both trees at part 0 and the prefix edges are
identical. A generated part matches a held-out part when part and color agree and its pose is
equivalent. Poses are compared through the world positions of the part's connectors (with their
types), which treats symmetric placements (e.g. a 2x2 plate turned 90 degrees) as the same.
Generated and held-out parts are paired one-to-one by maximum bipartite matching.
"""

from dataclasses import asdict, dataclass

import numpy as np
from bricknet import decode_graph, parse_sample, tree_to_graph
from bricknet.graph import _part_info
from scipy.optimize import linear_sum_assignment

# LDU. Parallel connections in the data disagree by up to ~2 LDU (e.g. val #417: a wheel rim placed via
# its axle vs. its hinge lands 1.8 LDU apart), while distinct placements differ by >= 4 LDU (half a plate).
TOL = 3.0


@dataclass(frozen=True)
class Score:
    parse_error: str | None
    n_generated: int  # parts after the prefix that parsed
    part_hits: int  # (part, color) multiset overlap with M, ignoring pose
    placed: int  # one-to-one matches on part, color and pose
    first_ok: bool  # the first generated part is a correct placement of some part of M
    exact: bool  # exactly M, every part correctly placed

    def asdict(self):
        return asdict(self)


def realize(text: str):
    """(parts, transforms, parse error) for the longest parsable head of a path text."""
    res = parse_sample(text)
    parts = res.tree.parts
    mats = decode_graph(tree_to_graph(res.tree)) if parts else []
    return parts, mats, res.error


def _signature(pid: int, T: np.ndarray):
    info = _part_info(pid)
    if len(info.frames) == 0:
        return T[None, :3, 3], np.zeros((1, 3))
    return (T @ info.frames)[:, :3, 3], info.meta[:, :3]


def same_pose(pid: int, Ta: np.ndarray, Tb: np.ndarray, tol: float = TOL) -> bool:
    pa, la = _signature(pid, Ta)
    pb, lb = _signature(pid, Tb)
    d = np.linalg.norm(pa[:, None] - pb[None], axis=-1)
    d[(la[:, None] != lb[None]).any(-1)] = np.inf  # only like connectors can stand in for each other
    return bool(d.min(1).max() <= tol and d.min(0).max() <= tol)


def score(prefix: str, target: str, completion: str, n_prefix: int, ref=None) -> Score:
    """Score one completion. ref: optional cached realize(prefix + target)."""
    r_parts, r_mats, r_err = ref or realize(prefix + target)
    assert r_err is None, r_err
    g_parts, g_mats, g_err = realize(prefix + completion)
    R = list(zip(r_parts[n_prefix:], r_mats[n_prefix:]))
    G = list(zip(g_parts[n_prefix:], g_mats[n_prefix:]))
    if g_err is None and len(g_parts) < n_prefix:
        g_err = "completion shorter than prefix"

    ok = np.zeros((len(G), len(R)), dtype=bool)
    for i, (gp, gm) in enumerate(G):
        for j, (rp, rm) in enumerate(R):
            ok[i, j] = gp == rp and same_pose(gp.part_id, gm, rm)
    placed = int(ok[linear_sum_assignment(-ok.astype(int))].sum()) if ok.size else 0

    want = {}
    for p, _ in R:
        want[p] = want.get(p, 0) + 1
    hits = 0
    for p, _ in G:
        if want.get(p, 0) > 0:
            want[p] -= 1
            hits += 1

    return Score(
        parse_error=g_err,
        n_generated=len(G),
        part_hits=hits,
        placed=placed,
        first_ok=bool(len(G) and ok[0].any()),
        exact=g_err is None and len(G) == len(R) and placed == len(R),
    )
