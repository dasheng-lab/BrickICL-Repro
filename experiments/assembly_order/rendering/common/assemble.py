"""Turn step 5 completion results into LDraw models. Shared by figures/ and benchmark/.

A test row gives the context (the parts of G \\ M, as path text) and the reference M. Each model wrote 8
completions. A "variant" of a row is one assembly to look at:

  context     G \\ M, what the model was shown
  reference   G \\ M + the true M
  <model>     G \\ M + what that model wrote (one chosen sample)

Every variant of a row is realized in the same frame, so the context parts sit at identical poses.
"""

import functools
import importlib.util
import json
import lzma
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bricknet import graph_to_ldr, parse_sample, tree_to_graph  # noqa: E402
from bricknet.graph import _part_info  # noqa: E402

from brickicl.eval.match import realize  # noqa: E402

MODELS = ("frozen", "plain", "masked")
VARIANTS = ("context", "reference", *MODELS)
CAPTIONS = REPO / "baseline/bricknet/data/dataset/captions_val.jsonl.xz"


def load_eval(eval_dir, models=MODELS):
    """(test rows by pair, {model: result rows by pair}) for the rows every model has."""
    eval_dir = Path(eval_dir)
    test = {r["pair"]: r for r in map(json.loads, open(eval_dir / "test.jsonl"))}
    runs = {m: {r["pair"]: r for r in map(json.loads, open(eval_dir / f"{m}.jsonl"))} for m in models}
    keep = [p for p in test if all(p in v for v in runs.values())]
    return {p: test[p] for p in keep}, runs


def load_captions():
    """Val captions; line i describes val model i, which is the `source` of a test row."""
    with lzma.open(CAPTIONS, "rt") as f:
        return {int(r["id"]): r["caption"] for r in map(json.loads, f)}


def case_id(pair):
    """'17:0' -> '17-0', the id the comparison viewer uses for the same row."""
    return pair.replace(":", "-")


def best_sample(samples):
    """Index of a model's best completion: exact first, then most parts placed, then most right parts."""
    return max(range(len(samples)), key=lambda i: (samples[i]["exact"], samples[i]["placed"], samples[i]["part_hits"], -i))


def variant_text(row, variant, result=None, sample=0):
    """Path text of one variant. result: that model's result row; sample: which of its completions."""
    if variant == "context":
        return row["prefix"]
    if variant == "reference":
        return row["prefix"] + row["target"]
    return row["prefix"] + result["samples"][sample]["text"]


def ldr_lines(text):
    """One LDraw type-1 line per part of the longest parsable head of a path text."""
    tree = parse_sample(text).tree
    if not tree.parts:
        return []
    return [ln for ln in graph_to_ldr(tree_to_graph(tree)).splitlines() if ln.startswith("1 ")]


@functools.cache
def _viewer_pack():
    """The comparison viewer's packer, which already knows how to judge each generated part."""
    spec = importlib.util.spec_from_file_location("viewer_pack", REPO / "experiments/assembly_order/viewer/pack.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verdicts(row, text):
    """Per generated part: 2 placed right, 1 right part and colour in the wrong place, 0 not a part of M."""
    flags, _ = _viewer_pack().verdicts(realize(row["prefix"] + row["target"]), row["n_prefix"], text)
    return flags


def recolor(lines, codes):
    """Replace the colour code of each line; None keeps the line's own colour."""
    out = []
    for ln, code in zip(lines, codes, strict=True):
        f = ln.split(maxsplit=2)
        out.append(ln if code is None else f"1 {code} {f[2]}")
    return out


def rotate(lines, yaw_deg):
    """Turn a model about the vertical (LDraw y) axis through the origin."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    out = []
    for ln in lines:
        f = ln.split(maxsplit=14)
        v = np.array(f[2:14], float)
        pos, mat = R @ v[:3], R @ v[3:].reshape(3, 3)
        out.append(" ".join(["1", f[1], *(f"{x:.6g}" for x in (*pos, *mat.ravel())), f[14]]))
    return out


def write_ldr(path, lines, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"0 {title}\n0 Name: {path.name}\n0 Author: brickICL experiments/assembly_order step 6\n" + "\n".join(lines) + "\n")


def extent_points(text):
    """World positions (LDU) that bound a model: every part's origin and connectors."""
    parts, mats, _ = realize(text)
    pts = []
    for p, T in zip(parts, mats):
        T = np.asarray(T, float)
        pts.append(T[None, :3, 3])
        frames = _part_info(p.part_id).frames
        if len(frames):
            pts.append((T @ frames)[:, :3, 3])
    return np.concatenate(pts)
