"""Pack probe inputs (context + candidate answers) and model samples for the web viewer.

The viewer (baseline/repro/viewer) tints every part from `heldout_from` on, so the held-out answer
stands out from the context the model actually sees, and the path panel marks where the answer starts.
For each source model one held-out copy (copy 0, walk seed 0) is shown.

Stretch probe (rows from brickicl.sampling.build_stretch), per model:
  1 original context + short copy    (the true answer)
  2 stretched context + long copy    (what following the context gives)
  3 stretched context + short copy   (what following the prior gives)
  4 ablated context + short copy
With --results (length_probe outputs, free or primed with --prime), also up to --samples model completions per
condition and file. For primed samples the given first part counts as context, so only the model's own parts
are tinted, and the note reports the generated leg length against the short and long copies.

Pilot (rows from brickicl.sampling.build): evidence and ablated context + the held-out copy.

Usage (repo root):
  python -m brickicl.viz.pack_probe stretch.jsonl [--results stretch.0.6B.jsonl stretch.p1.1.7B.jsonl --samples 3]
  python -m brickicl.viz.pack_probe pilot.jsonl --kind pilot
Then serve baseline/repro/viewer (the `bricknet-viewer` launch config, port 8765).
"""

import argparse
import importlib.util
import json
import lzma
import re
from pathlib import Path

from bricknet import graph_to_ldr, parse_sample, tree_to_graph
from brickicl.viz.lazy_catalog import write_catalog

REPO = Path(__file__).resolve().parents[2]
VIEWER = REPO / "baseline/repro/viewer"
CAPTIONS = REPO / "baseline/bricknet/data/dataset/captions_val.jsonl.xz"


def viewer_pack():
    spec = importlib.util.spec_from_file_location("viewer_pack", VIEWER / "pack.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STRETCH_VIEWS = [  # (condition, answer, title, note)
    ("original", "short", "① 原样上下文 + 短腿（真实答案）",
     "其他副本保持原样，<b>答案</b>是被挖掉的那一份原样副本。"),
    ("stretched", "long", "② 加长上下文 + 长腿（跟随上下文）",
     "其他副本都加长了 {r} 格；<b>答案</b>是同样加长的这一份，即跟随上下文时应生成的结果。"),
    ("stretched", "short", "③ 加长上下文 + 短腿（跟随先验）",
     "上下文与 ② 相同，<b>答案</b>换成原样长度，即只靠训练先验时会生成的结果。"),
    ("ablated", "short", "④ 删除上下文 + 短腿",
     "其他副本全部删除，上下文里没有可参照的副本；<b>答案</b>是原样副本。"),
]
PILOT_VIEWS = [
    ("evidence", "① 有证据：其他副本在上下文里", "其他副本都在上下文里，<b>答案</b>是被挖掉的这一份。"),
    ("ablated", "② 无证据：其他副本被删除", "其他副本全部删除，<b>答案</b>不变。"),
]


def first_sentence(text, limit=90):
    s = re.split(r"(?<=\.)\s", text.strip(), maxsplit=1)[0]
    return s if len(s) <= limit else s[: limit - 1] + "…"


class Packer:
    def __init__(self):
        self.pack = viewer_pack()
        self.index = self.pack.build_index()
        (VIEWER / "models").mkdir(exist_ok=True)
        if not (VIEWER / "colors.ldr").exists():
            self.pack.write_colors()
        self.entries = []

    def add(self, name, text, heldout_from, **meta):
        res = parse_sample(text)
        if not res.tree.parts:
            print(f"skip {name}: nothing parsable ({res.error})")
            return
        lines = [ln for ln in graph_to_ldr(tree_to_graph(res.tree)).splitlines() if ln.startswith("1 ")]
        mpd, missing = self.pack.embed(lines, self.index)
        if missing:
            print(f"{name}: {len(missing)} sub-files not in the library")
        rel = f"single_model/{name}" if "/" not in name else name
        dest = VIEWER / "models" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(mpd)
        self.entries.append(meta | {
            "task": "single_model",
            "file": rel,
            "path": text,
            "heldout_from": heldout_from,
            "parse_error": res.error,
            "n_parts": len(lines),
            "size_kb": round(len(mpd) / 1024),
        })

    def add_pair(self, name, demo_text, query_text, query_prefix_parts, **meta):
        """Pack two independent paths as left/right models with one build timeline."""
        demo, query = parse_sample(demo_text), parse_sample(query_text)
        if demo.error is not None or not demo.tree.parts:
            raise ValueError(f"invalid complete demonstration: {demo.error}")
        if len(query.tree.parts) < query_prefix_parts:
            raise ValueError(f"query lost its given prefix: {query.error}")
        demo_lines = [ln for ln in graph_to_ldr(tree_to_graph(demo.tree)).splitlines() if ln.startswith("1 ")]
        query_lines = [ln for ln in graph_to_ldr(tree_to_graph(query.tree)).splitlines() if ln.startswith("1 ")]

        def x(line):
            return float(line.split(maxsplit=14)[2])

        shift = max(map(x, demo_lines)) - min(map(x, query_lines)) + 160.0
        shifted = []
        for line in query_lines:
            fields = line.split(maxsplit=14)
            fields[2] = f"{float(fields[2]) + shift:.3f}"
            shifted.append(" ".join(fields))
        lines = demo_lines + shifted
        mpd, missing = self.pack.embed(lines, self.index)
        if missing:
            print(f"{name}: {len(missing)} sub-files not in the library")
        rel = f"two_copy/{name}"
        dest = VIEWER / "models" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(mpd)
        self.entries.append(meta | {
            "task": "two_copy",
            "file": rel,
            "demo_path": demo_text,
            "query_path": query_text,
            "demo_parts": len(demo_lines),
            "query_prefix_parts": query_prefix_parts,
            "heldout_from": len(demo_lines) + query_prefix_parts,
            "parse_error": query.error,
            "n_parts": len(lines),
            "size_kb": round(len(mpd) / 1024),
        })

    def save(self):
        path = VIEWER / "models/manifest.json"
        old = json.loads(path.read_text()) if path.exists() else []
        new = {e["file"] for e in self.entries}
        manifest = self.entries + [e for e in old if e["file"] not in new]  # probe entries first
        path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        write_catalog(VIEWER / "models", manifest)
        print(f"packed {len(self.entries)} views into {VIEWER / 'models'} (manifest: {len(manifest)} entries)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="build_stretch or build output jsonl")
    ap.add_argument("--kind", choices=["stretch", "pilot"], default="stretch")
    ap.add_argument("--results", type=Path, nargs="*", default=[],
                    help="length_probe outputs for the same data (adds model samples; free or primed, any model)")
    ap.add_argument("--samples", type=int, default=3, help="model samples per condition (with --results)")
    ap.add_argument("--sources", type=int, nargs="*", help="only these model ids")
    args = ap.parse_args()

    captions = [json.loads(line)["caption"] for line in lzma.open(CAPTIONS, "rt")]
    rows = [json.loads(line) for line in open(args.data)]
    first = {}
    for r in rows:  # one held-out copy per model: copy 0, seed 0
        if r["pair"].endswith("-0-0") and (not args.sources or r["source"] in args.sources):
            first[(r["source"], r["condition"])] = r
    results_by_file = [{(r["pair"], r["condition"]): r for r in map(json.loads, open(path))} for path in args.results]

    packer = Packer()
    for src in sorted({s for s, _ in first}):
        cap = first_sentence(captions[src])
        if args.kind == "stretch":
            r0 = first[(src, "original")]
            layout = {"contiguous": "连续", "scattered": "打散"}.get(r0.get("layout", "scattered"))
            group = f"长腿测试 r={r0['r']}（{layout}）· #{src}"
            for i, (cond, ans, title, note) in enumerate(STRETCH_VIEWS):
                r = first[(src, cond)]
                packer.add(f"stretch_r{r['r']}_{src}_{i + 1}.mpd", r["prefix"] + r[f"target_{ans}"], r["n_prefix"],
                           group=group, title=title, caption=cap, note=note.format(r=r["r"]))
            for results in results_by_file:
                for cond in ("original", "stretched", "ablated"):
                    res = results.get((first[(src, cond)]["pair"], cond))
                    if not res:
                        continue
                    tag = {"original": "原样", "stretched": "加长", "ablated": "删除"}[cond]
                    size = res["model"].split("-")[-1]
                    prime = res.get("prime", 0)
                    for j, s in enumerate(res["samples"][: args.samples]):
                        name = f"stretch_r{res['r']}_{src}_{cond}_{size}{f'_p{prime}' if prime else ''}_s{j}.mpd"
                        text = first[(src, cond)]["prefix"] + s["text"]
                        if prime:  # the given first part counts as context; only the model's own parts are tinted
                            leg = ("长腿" if s["leg"] == res["leg_long"] else "短腿" if s["leg"] == res["leg_short"]
                                   else "长于长腿" if s["leg"] > res["leg_long"] else "短于短腿" if s["leg"] < res["leg_short"]
                                   else "介于两者之间")
                            exact = {"long": "，而且整条长腿完全正确", "short": "，而且整条短腿完全正确"}.get(s["category"], "")
                            packer.add(name, text, res["n_prefix"] + prime, group=group, caption=cap,
                                       title=f"{tag}上下文 · {size} 给开头 · 样本 {j + 1}",
                                       note=f"提示里给了缺腿的第一块零件（青绿色前的最后一块）。<b>模型生成</b>的腿上有 "
                                            f"{s['leg']} 块被加长的那种零件（短腿 {res['leg_short']} 块，长腿 {res['leg_long']} 块）"
                                            f"，长度{'等于' if leg in ('长腿', '短腿') else ''}{leg}{exact}；共生成 {s['n_parts'] - prime} 件。")
                        else:
                            verdict = {"long": "生成了长腿", "short": "生成了短腿"}.get(s["category"], "既不是长腿也不是短腿")
                            packer.add(name, text, res["n_prefix"], group=group, caption=cap,
                                       title=f"{tag}上下文 · {size} 样本 {j + 1}",
                                       note=f"<b>模型生成</b>的部分：{verdict}，共 {s['n_parts']} 件，其中被加长的那种零件 "
                                            f"{s['n_stack']} 件（短腿 {res['k_short']} 件，长腿 {res['k_long']} 件）。")
        else:
            group = f"挖空试点 · #{src}"
            for i, (cond, title, note) in enumerate(PILOT_VIEWS):
                r = first[(src, cond)]
                packer.add(f"pilot_{src}_{i + 1}.mpd", r["prefix"] + r["target"], r["n_prefix"],
                           group=group, title=title, caption=cap, note=note)
    packer.save()


if __name__ == "__main__":
    main()
