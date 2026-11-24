"""Make a condition-balanced, source-preserving training subset from train.jsonl."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def rank(pair):
    return hashlib.sha256(f"balance:{pair}".encode()).digest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    groups = defaultdict(dict)
    with (args.data / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            groups[row["pair"]][row["condition"]] = row
    triples = [pair for pair, g in groups.items() if {"original", "deleted", "stretched"} <= g.keys()]
    if not triples:
        raise ValueError("no complete original/deleted/stretched triples")
    triple_set = set(triples)
    selected = [groups[pair][condition] for pair in sorted(triples, key=rank)
                for condition in ("original", "deleted", "stretched")]
    extra = [pair for pair, g in groups.items() if pair not in triple_set and {"original", "deleted"} <= g.keys()]
    for pair in sorted(extra, key=rank)[:len(triples)]:
        selected.extend((groups[pair]["original"], groups[pair]["deleted"]))
    random.Random(42).shuffle(selected)
    with (args.data / "train_balanced.jsonl").open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(row["condition"] for row in selected)
    metadata_path = args.data / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["balanced_train_counts"] = dict(counts)
    metadata["balanced_train_rows"] = len(selected)
    metadata["balanced_train_sources"] = len({row["source"] for row in selected})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"balanced {len(selected)} unique rows from {metadata['balanced_train_sources']} sources: {dict(counts)}")


if __name__ == "__main__":
    main()
