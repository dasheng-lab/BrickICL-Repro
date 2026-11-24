"""Publish only matched G -> G/M views, with a shared visible build path."""
import argparse
import json
import lzma
from pathlib import Path

from bricknet import parse_sample
from brickicl.viz.pack_probe import CAPTIONS, Packer, first_sentence


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(row):
    demo, query = row["demo_text"], row["query_prefix"]
    expected = row["target_long" if row["condition"] == "stretched" else "target_short"]
    if demo != query + expected:
        raise ValueError(f"different known assemblies: {row['pair']} {row['condition']}")
    for label, text in (("demo", demo), ("query", query)):
        parsed = parse_sample(text)
        if parsed.error:
            raise ValueError(f"{row['pair']} {row['condition']} {label}: {parsed.error}")
    if row["n_prefix"] != len(parse_sample(query).tree.parts):
        raise ValueError(f"bad query prefix count: {row['pair']} {row['condition']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--sources", type=int, nargs="*")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    rows = {(r["pair"], r["condition"]): r for r in read_jsonl(args.data)}
    for row in rows.values():
        validate(row)
    captions = [json.loads(line)["caption"] for line in lzma.open(CAPTIONS, "rt")]
    packer = Packer()
    references = set()
    for path in args.results:
        for result in read_jsonl(path):
            if args.sources and result["source"] not in args.sources:
                continue
            key = result["pair"], result["condition"]
            row = rows[key]
            source, condition = result["source"], result["condition"]
            walk_seed = result.get("walk_seed_index", row.get("walk_seed_index", 0))
            size = result["model"].split("-")[-1]
            tag = {"original": "原样", "stretched": "加长", "ablated": "删除"}[condition]
            group = f"同构双模型 r={result['r']} · walk seed {walk_seed} · #{source}"
            caption = first_sentence(captions[source])
            # Preserve seed 0 names so rebuilding replaces legacy single-seed Viewer entries.
            seed_tag = "" if walk_seed == 0 else f"_w{walk_seed}"
            stem = f"matched_{source}{seed_tag}_{condition}"
            if key not in references:
                packer.add_pair(
                    f"{stem}_reference.mpd", row["demo_text"], row["demo_text"],
                    row["n_prefix"], group=group, caption=caption,
                    title=f"{tag} · 完整答案", note="左右两侧是同一个组装路径；右侧只给到目标腿的第一块。",
                )
                references.add(key)
            for j, sample in enumerate(result["samples"][:args.samples], 1):
                completion = sample["raw_completion"]
                query = row["query_prefix"] + completion
                packer.add_pair(
                    f"{stem}_{size}_sample{j}.mpd", row["demo_text"], query,
                    row["n_prefix"], group=group, caption=caption,
                    title=f"{tag} · {size} 样本 {j}",
                    note=(f"左侧完整示范与右侧给定前缀逐件一致；右侧后续由 {size} 生成。"
                          f"目标腿零件 {sample['leg']} 件；参考短腿 {result['leg_short']} 件，"
                          f"参考长腿 {result['leg_long']} 件；解析错误：{sample['parse_error'] or '无'}。"),
                )
    packer.save()


if __name__ == "__main__":
    main()
