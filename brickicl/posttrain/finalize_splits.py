"""Source-family split for post-training; keep previously explored val as regression only."""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from brickicl.posttrain.build_dataset import fingerprint
from brickicl.sampling.build import read_graphs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sft-graphs", required=True)
    args = ap.parse_args()
    meta_path = args.data / "metadata.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("source_family_split_finalized"):
        raise RuntimeError("already finalized; refusing to repartition")
    graphs = read_graphs(args.sft_graphs)
    source_to_family, family_to_split = {}, {}
    counts, sources = Counter(), defaultdict(set)
    streams = {split: (args.data / f"{split}.split.tmp").open("w", encoding="utf-8")
               for split in ("train", "dev", "test")}
    try:
        for line in (args.data / "train.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            source = row["source"]
            if source not in source_to_family:
                family = fingerprint(graphs[source])
                source_to_family[source] = family
                bucket = int(family[:12], 16) % 20
                family_to_split[family] = "test" if bucket == 0 else "dev" if bucket == 1 else "train"
            family = source_to_family[source]
            split = family_to_split[family]
            row.update(split=split, source_family=family)
            row["pair"] = f"{split}-{source}-{row['target_copy']}-{row['walk_seed']}"
            streams[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
            counts[f"{split}:{row['condition']}"] += 1
            sources[split].add(source)
    finally:
        for stream in streams.values():
            stream.close()
    regression = args.data / "val_regression.jsonl"
    with regression.open("w", encoding="utf-8") as dest:
        for split in ("dev", "test"):
            for line in (args.data / f"{split}.jsonl").open(encoding="utf-8"):
                row = json.loads(line)
                row["split"] = "val_regression"
                row["pair"] = f"val_regression-{row['source']}-{row['target_copy']}-{row['walk_seed']}"
                dest.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["val_regression"] += 1
                counts[f"val_regression:{row['condition']}"] += 1
                sources["val_regression"].add(row["source"])
    for split in streams:
        os.replace(args.data / f"{split}.split.tmp", args.data / f"{split}.jsonl")
    manifest = [{"source_split": "sft", "source": source, "family": family,
                 "split": family_to_split[family]} for source, family in sorted(source_to_family.items())]
    (args.data / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    meta.update(source_family_split_finalized=True,
                upstream_splits={"train": "sft", "dev": "sft", "test": "sft", "val_regression": "val"},
                split_rule="color-independent source-family fingerprint bucket modulo 20: 0 test, 1 dev, 2..19 train",
                counts=dict(counts), source_counts={k: len(v) for k, v in sources.items()},
                limitations="train/dev/test measure post-training generalization on disjoint source families. Their SFT sources were available to upstream BrickNet PT training. Official val repeated structures have already been explored and are retained only as val_regression. Conservative fingerprint screening does not establish absence of every near duplicate.")
    meta["historical_val_longleg_sources"] = meta.pop("known_val_sources_forced_to_dev", [])
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "sources": meta["source_counts"]}, indent=2))


if __name__ == "__main__":
    main()
