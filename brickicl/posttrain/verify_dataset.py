"""Check generated single-path ICL rows and split isolation without loading models."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from bricknet import parse_sample


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    metadata = json.loads((args.data / "metadata.json").read_text())
    sources, counts, digests = defaultdict(set), Counter(), set()
    checksums = {}
    families = defaultdict(set)
    for split in ("train", "dev", "test"):
        path = args.data / f"{split}.jsonl"
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for line in f:
                sha.update(line)
                row = json.loads(line)
                assert row["split"] == split
                allowed = metadata["upstream_splits"][split]
                assert row["source_split"] in ([allowed] if isinstance(allowed, str) else allowed)
                if "source_family" in row:
                    families[split].add(row["source_family"])
                assert len(row["prompt"].splitlines()) == 2 * row["n_prefix"] - 1
                assert len(row["completion"].splitlines()) == 2 * row["k"]
                assert row["tokens_with_eos"] <= metadata["max_tokens_with_eos"]
                digest = hashlib.sha256((row["prompt"] + row["completion"]).encode()).digest()
                assert digest not in digests, "duplicate full path across rows"
                digests.add(digest)
                parsed = parse_sample(row["prompt"] + row["completion"])
                assert parsed.error is None, (split, row["source"], parsed.error)
                assert len(parsed.tree.parts) == row["n_prefix"] + row["k"]
                sources[split].add((row["source_split"], row["source"]))
                if metadata.get("version") == 2:
                    assert row["k"] >= 8 and row["target_tokens_with_eos"] >= 192
                    assert row["target_fraction_parts"] >= .2 and row["loss_token_fraction"] >= .2
                    assert row["evidence_copy_count"] >= 1
                    assert all(i < row["n_prefix"] for i in row["evidence_positions"])
                    assert len(row["evidence_positions"]) == row["evidence_copy_count"] * row["motif_size"]
                if metadata.get("version") == 3:
                    assert 6 <= row['k'] <= 10 and row['target_copy_count'] == 1
                    assert 50 <= row['n_prefix'] + row['k'] <= 160
                    assert row['target_fraction_parts'] <= .18
                    assert row['body_parts'] >= 30 and row['body_part_types'] >= 5
                    assert row['body_max_part_type_fraction'] <= .45
                    assert row['evidence_copy_count'] >= 1
                    assert all(i < row['n_prefix'] for i in row['evidence_positions'])
                    assert len(row['evidence_positions']) == row['evidence_copy_count'] * row['k']
                    if row['layout'] == 'distant': assert row['evidence_gap_parts'] >= 10
                counts[split] += 1
                counts[f"{split}:{row['condition']}"] += 1
        checksums[split] = sha.hexdigest()
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not sources[a] & sources[b]
        assert not families[a] & families[b]
    assert all(counts[split] == metadata["counts"].get(split, 0) for split in sources)
    report = {"verified_counts": dict(counts), "source_counts": {k: len(v) for k, v in sources.items()},
              "family_counts": {k: len(v) for k, v in families.items()}, "split_overlap": 0,
              "sha256": checksums}
    (args.data / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
