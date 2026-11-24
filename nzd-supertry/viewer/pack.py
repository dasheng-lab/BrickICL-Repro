"""Pack the step 5 completion results for the comparison viewer (index.html in this folder).

Reads test.jsonl and one result file per model from step5_eval/eval_completion.py and writes, into --out:

  index.json        the columns, the overall numbers, one light record per test row (for the folder tree)
                    and the dataset's caption of every val model that has a test row
  cases/<id>.json   one test row: the prefix, the reference M, and every sample of every model as LDraw
                    lines with a per-part verdict (placed right / right part, wrong place / wrong part)
  lib/<part>.json   an LDraw part with every sub-file it references, fetched by the page on demand
  colors.ldr        LDraw colours, including BrickNet's extended codes

Everything here derives from the gated dataset: --out is not tracked by git.

Usage (repo root, bricknet env):
  python nzd-supertry/viewer/pack.py runs/nzd-supertry/eval_v2
"""

import argparse
import importlib.util
import json
import lzma
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from bricknet import graph_to_ldr, parse_sample, tree_to_graph  # noqa: E402

from brickicl.eval.match import realize, same_pose  # noqa: E402

COLUMNS = [
    {"id": "ref", "name": "参考答案", "sub": "数据集里真实的 M", "kind": "ref"},
    {"id": "frozen", "name": "冻结 PT", "sub": "官方 BrickNet-0.6B-PT，不训练", "kind": "model"},
    {"id": "plain", "name": "对照", "sub": "深度优先顺序后训练", "kind": "model"},
    {"id": "masked", "name": "增强", "sub": "掩码顺序后训练", "kind": "model"},
]
METRICS = ("exact", "place_acc", "first_ok", "part_acc", "anchor_ref", "on_path", "parse_ok")


def load_captions():
    """The dataset's one-sentence description of each val model; line i describes val model i, a row's `source`."""
    with lzma.open(REPO / "baseline/bricknet/data/dataset/captions_val.jsonl.xz", "rt") as f:
        return {int(r["id"]): r["caption"] for r in map(json.loads, f)}


def viewer_pack():
    """The existing viewer's packer: LDraw library index and colour table."""
    spec = importlib.util.spec_from_file_location("viewer_pack", REPO / "baseline/repro/viewer/pack.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ldr_lines(text):
    """One LDraw type-1 line per part of the longest parsable head of a path text."""
    tree = parse_sample(text).tree
    if not tree.parts:
        return []
    return [ln for ln in graph_to_ldr(tree_to_graph(tree)).splitlines() if ln.startswith("1 ")]


def part_texts(text):
    """[node line, edge line] per part of a completion (the root of a full path has no edge line)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [lines[i : i + 2] for i in range(0, len(lines), 2)]


def verdicts(ref, n0, text):
    """Per generated part: 2 placed right (one-to-one with M), 1 right part and colour only, 0 neither.

    Also the parts of M that were reproduced, so the page can outline the ones still missing.
    """
    r_parts, r_mats, _ = ref
    g_parts, g_mats, _ = realize(text)
    R = list(zip(r_parts[n0:], r_mats[n0:]))
    G = list(zip(g_parts[n0:], g_mats[n0:]))
    ok = np.zeros((len(G), len(R)), dtype=bool)
    for i, (gp, gm) in enumerate(G):
        for j, (rp, rm) in enumerate(R):
            ok[i, j] = gp == rp and same_pose(gp.part_id, gm, rm)
    out, hit = [0] * len(G), []
    if ok.size:
        for i, j in zip(*linear_sum_assignment(-ok.astype(int))):
            if ok[i, j]:
                out[i] = 2
                hit.append(int(j))
    want = {}
    for p, _ in R:
        want[p] = want.get(p, 0) + 1
    for i, (p, _) in enumerate(G):  # the same greedy multiset count as match.score
        if want.get(p, 0) > 0:
            want[p] -= 1
            out[i] = max(out[i], 1)
    return out, sorted(hit)  # hit: which parts of M (0..k-1) some generated part reproduces


def summarize(rows):
    out = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
    out["nll_M"] = float(np.mean([-r["lp_target"] / r["n_target_tokens"] for r in rows]))
    out["eos_loss"] = float(np.mean([-r["lp_eos"] for r in rows]))
    out["any_exact"] = float(np.mean([any(s["exact"] for s in r["samples"]) for r in rows]))
    out["any_first"] = float(np.mean([any(s["first_ok"] for s in r["samples"]) for r in rows]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path, help="directory with test.jsonl and <model>.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--models", nargs="+", default=["frozen", "plain", "masked"])
    ap.add_argument("--limit", type=int, default=None, help="first N test rows only")
    ap.add_argument("--title", default="掩码顺序增强 · val 补全对比")
    args = ap.parse_args()

    test = {}
    for line in open(args.eval_dir / "test.jsonl"):
        r = json.loads(line)
        test[r["pair"]] = r
    runs = {}
    for name in args.models:
        runs[name] = {r["pair"]: r for r in map(json.loads, open(args.eval_dir / f"{name}.jsonl"))}
    pairs = [p for p in test if all(p in v for v in runs.values())][: args.limit]
    print(f"{len(pairs)} test rows, models {args.models}")

    vp = viewer_pack()
    lib_index = vp.build_index()
    (args.out / "cases").mkdir(parents=True, exist_ok=True)
    (args.out / "lib").mkdir(exist_ok=True)
    colors = REPO / "baseline/repro/viewer/colors.ldr"
    if not colors.exists():
        vp.write_colors()
    shutil.copyfile(colors, args.out / "colors.ldr")

    needed, cases = set(), []
    for n_done, pair in enumerate(pairs):
        t = test[pair]
        n0, k = t["n_prefix"], t["k"]
        full = t["prefix"] + t["target"]
        ref = realize(full)
        ref_lines = ldr_lines(full)
        assert len(ref_lines) == n0 + k, (pair, len(ref_lines), n0, k)
        cid = pair.replace(":", "-")
        case = {
            "id": cid, "source": t["source"], "sample_index": t["sample_index"], "k": k, "n_prefix": n0,
            "nodes": t["nodes"], "jump": t["jump"], "lag": t["lag"],
            "prefix": {"lines": ref_lines[:n0], "text": [[t["prefix"].split("\n")[0]]] + part_texts(
                t["prefix"].split("\n", 1)[1])},
            "ref": {"lines": ref_lines[n0:], "text": part_texts(t["target"])},
            "models": {},
        }
        light = {key: case[key] for key in ("id", "source", "sample_index", "k", "n_prefix", "jump", "lag")}
        light["s"] = {}
        for name in args.models:
            r = runs[name][pair]
            case["ref"]["anchor"] = r["ref_anchor"]
            samples = []
            for s in r["samples"]:
                lines = ldr_lines(t["prefix"] + s["text"])[n0:]
                flags, hit = verdicts(ref, n0, t["prefix"] + s["text"])
                assert len(flags) == len(lines) and len(hit) == s["placed"], (pair, name)
                samples.append({
                    "lines": lines, "flags": flags, "hit": hit, "text": part_texts(s["text"]),
                    "exact": s["exact"], "placed": s["placed"], "part_hits": s["part_hits"],
                    "first_ok": s["first_ok"], "anchor": s["first_anchor"], "anchor_ref": s["anchor_ref"],
                    "on_path": s["on_path"], "parse_error": s["parse_error"],
                })
            case["models"][name] = {
                "nll_M": -r["lp_target"] / r["n_target_tokens"], "eos_loss": -r["lp_eos"],
                **{m: r[m] for m in METRICS}, "samples": samples,
            }
            light["s"][name] = [round(r["exact"], 4), round(r["place_acc"], 4), round(r["first_ok"], 4),
                                int(any(s["exact"] for s in r["samples"]))]
        refs = {ln.split(maxsplit=14)[14].strip().replace("\\", "/").lower()
                for ln in ref_lines + [ln for m in case["models"].values() for s in m["samples"] for ln in s["lines"]]}
        case["parts"] = sorted(refs)
        needed |= refs
        (args.out / "cases" / f"{cid}.json").write_text(json.dumps(case, ensure_ascii=False, separators=(",", ":")))
        cases.append(light)
        if (n_done + 1) % 200 == 0:
            print(f"  {n_done + 1}/{len(pairs)} rows", flush=True)

    missing = []
    for ref_name in sorted(needed):
        sections, queue = {}, [ref_name]
        while queue:
            name = queue.pop()
            key = vp.loader_key(name)
            if key in sections:
                continue
            f = lib_index.get(name.replace("\\", "/").lower())
            if f is None:
                missing.append(name)
                sections[key] = ""
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            sections[key] = text
            queue.extend(vp.sub_refs(text))
        (args.out / "lib" / (re.sub(r"[^\w.-]", "_", ref_name) + ".json")).write_text(
            json.dumps(sections, separators=(",", ":")))
    if missing:
        print(f"{len(set(missing))} referenced files are not in the LDraw library: {sorted(set(missing))[:8]}")

    subsets = [("all", "全部题目", lambda r: True), ("jump", "需要跳接", lambda r: r["jump"]),
               ("nojump", "不需要跳接", lambda r: not r["jump"])]
    overview = []
    for sid, label, keep in subsets:
        sub = [p for p in pairs if keep(test[p])]
        overview.append({"id": sid, "name": label, "rows": len(sub), "models_n": len({test[p]["source"] for p in sub}),
                         "models": {name: summarize([runs[name][p] for p in sub]) for name in args.models}})
    head = next(iter(runs[args.models[0]].values()))
    index = {
        "title": args.title,
        "columns": [c for c in COLUMNS if c["id"] == "ref" or c["id"] in args.models],
        "adapters": {name: next(iter(runs[name].values()))["lora"] for name in args.models},
        "n_samples": head["n"],
        "overview": overview,
        "captions": {str(src): text for src, text in sorted(load_captions().items()) if src in {c["source"] for c in cases}},
        "cases": cases,
    }
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {len(cases)} cases, {len(needed)} part bundles -> {args.out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
