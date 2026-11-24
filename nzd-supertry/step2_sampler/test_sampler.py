"""Tests for the masked-order sampler.

The first group runs on the car bundled with BrickNet and needs no dataset. The second group uses
val.npz.xz and is skipped when the gated dataset is not there. Both groups need the collider
meshes under $BRICKNET_DATA/cvx (`python -m bricknet fetch-meshes`): the walk checks collisions.

  python nzd-supertry/step2_sampler/test_sampler.py -v
"""

import io
import lzma
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("BRICKNET_DATA", str(REPO / "baseline/bricknet/data"))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(REPO)]

import bricknet  # noqa: E402
from bricknet import decode_graph, load_graphs, parse_ldr, parse_sample, score_text, tree_to_graph  # noqa: E402
from sampler import (  # noqa: E402
    choose_block,
    collision_free_walk,
    is_connected,
    n_pairs_for,
    preorder,
    sample_pairs,
    serializable,
    simple_adj,
    subtree_sizes,
)

VAL = Path(os.environ["BRICKNET_DATA"]) / "dataset" / "val.npz.xz"


def car():
    return parse_ldr((Path(bricknet.__file__).parent / "_data" / "car.ldr").read_text())


def parents(text):
    """Parent position of every part after the first, from the path text."""
    return [e.parent for e in parse_sample(text).tree.edges]


def off_path(text):
    """Positions whose parent is neither the previous part nor one of its ancestors.
    A depth-first preorder has none."""
    par = [None] + parents(text)
    out = []
    for i in range(2, len(par)):
        u, ok = i - 1, False
        while u is not None:
            if u == par[i]:
                ok = True
                break
            u = par[u]
        if not ok:
            out.append(i)
    return out


def check_pair(test, graph, pair):
    plain, masked = pair["plain"], pair["masked"]
    pre, moved, M = plain["order"], masked["order"], masked["M"]
    n, k, i = plain["nodes"], masked["k"], plain["block_start"]
    adj = simple_adj(graph, set(pre))

    # the masked order is the plain order with one contiguous block moved to the end
    test.assertEqual(pre[i : i + k], M)
    test.assertEqual(moved, pre[:i] + pre[i + k :] + M)
    test.assertEqual(masked["n_prefix"], n - k)
    test.assertLess(i + k, n, "the block was already last")
    test.assertTrue(1 <= k <= n // 2 and n - k >= 2)

    # M is a valid hole: connected, and the rest stays connected
    test.assertTrue(is_connected(M, adj))
    test.assertTrue(is_connected(set(pre) - set(M), adj))

    # both texts parse; the prefix alone is a complete walk of the context
    for row in (plain, masked):
        res = parse_sample(row["path"])
        test.assertIsNone(res.error)
        test.assertEqual(len(res.tree.parts), n)
        test.assertTrue(row["path"].endswith("\n"))
    lines = masked["path"].splitlines(keepends=True)
    res = parse_sample("".join(lines[: 1 + 2 * (n - k - 1)]))
    test.assertIsNone(res.error)
    test.assertEqual(len(res.tree.parts), n - k)

    # plain is a depth-first preorder; masked breaks it at most once, where M starts (`jump`)
    test.assertEqual(off_path(plain["path"]), [])
    test.assertEqual(off_path(masked["path"]), [n - k] if masked["jump"] else [])

    # `lag` is the distance from the first part of M back to the part it attaches to
    test.assertEqual(masked["lag"], (n - k) - parents(masked["path"])[n - k - 1])
    test.assertEqual(plain["lag"], i - parents(plain["path"])[i - 1])
    test.assertGreater(masked["lag"], plain["lag"])

    # same tree, so the same geometry: every part has the same pose relative to the root
    a = dict(zip(pre, decode_graph(tree_to_graph(parse_sample(plain["path"]).tree))))
    b = dict(zip(moved, decode_graph(tree_to_graph(parse_sample(masked["path"]).tree))))
    test.assertEqual(pre[0], moved[0])
    test.assertLess(max(np.abs(a[v] - b[v]).max() for v in pre), 1e-6)

    # collision-free under the scorer (None: a collider mesh is missing)
    test.assertFalse(score_text(plain["path"])[2])
    test.assertFalse(score_text(masked["path"])[2])


class CarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = car()

    def test_preorder_and_blocks(self):
        order, parent_of, _ = collision_free_walk(self.g, seed=0)
        rng = np.random.default_rng(0)
        pre = preorder(order, parent_of, rng)
        self.assertEqual(sorted(pre), sorted(order))
        self.assertEqual(pre[0], order[0])
        pos = {v: i for i, v in enumerate(pre)}
        self.assertTrue(all(pos[parent_of[v][0]] < pos[v] for v in pre[1:]))
        size = subtree_sizes(pre, parent_of)
        self.assertEqual(size[pre[0]], len(pre))
        for i, v in enumerate(pre):  # the subtree of pre[i] is the block pre[i : i + size]
            block = set(pre[i : i + size[v]])
            self.assertTrue(all(parent_of[u][0] in block for u in block - {v}))
            self.assertTrue(all(parent_of[u][0] not in block or u in block for u in pre[1:]))
        sizes = set()
        for _ in range(200):
            i = choose_block(pre, size, rng, k_max=len(pre) // 2)
            self.assertIsNotNone(i)
            self.assertLessEqual(size[pre[i]], len(pre) // 2)
            self.assertLess(i + size[pre[i]], len(pre))
            sizes.add(size[pre[i]])
        self.assertGreater(len(sizes), 2, "M sizes should vary")
        self.assertNotEqual(pre, preorder(order, parent_of, np.random.default_rng(1)), "sibling order should vary")

    def test_pairs(self):
        pairs, stats = sample_pairs(self.g, 6, seed=1)
        self.assertGreater(len(pairs), 0)
        self.assertEqual(stats["pairs"], len(pairs))
        for pair in pairs:
            check_pair(self, self.g, pair)
        self.assertEqual(len({p["plain"]["path"] for p in pairs}), len(pairs), "plain text repeats")
        self.assertEqual(len({p["masked"]["path"] for p in pairs}), len(pairs), "masked text repeats")

    def test_deterministic(self):
        a, _ = sample_pairs(self.g, 4, seed=7)
        b, _ = sample_pairs(self.g, 4, seed=7)
        c, _ = sample_pairs(self.g, 4, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual([p["masked"]["path"] for p in a], [p["masked"]["path"] for p in c])

    def test_walk_matches_upstream(self):
        order, parent_of, text = collision_free_walk(self.g, seed=3)
        tree = bricknet.graph.sample_collision_free_tree(self.g, seed=3)
        self.assertEqual(text, bricknet.serialize_tree(tree))
        self.assertEqual(len(order), len(tree.parts))
        self.assertEqual(set(parent_of), set(order[1:]))

    def test_n_pairs(self):
        self.assertEqual([n_pairs_for(n) for n in (2, 6, 18, 40, 100, 400)], [1, 2, 4, 10, 25, 25])


@unittest.skipUnless(VAL.exists(), "gated dataset not downloaded")
class ValTests(unittest.TestCase):
    def test_pairs_on_val(self):
        graphs = load_graphs(io.BytesIO(lzma.open(VAL).read()))
        rng = np.random.default_rng(0)
        n_pairs = n_models = 0
        for gid in sorted(rng.choice(len(graphs), size=40, replace=False).tolist()):
            g = graphs[gid]
            if len(g.part_ids) < 6 or not serializable(g):
                continue
            pairs, _ = sample_pairs(g, n_pairs_for(len(g.part_ids)), seed=gid)
            for pair in pairs:
                check_pair(self, g, pair)
            n_pairs += len(pairs)
            n_models += bool(pairs)
        self.assertGreater(n_models, 20)
        self.assertGreater(n_pairs, 100)


if __name__ == "__main__":
    unittest.main()
