"""Remove rows with missing collision meshes; run after build and before balance."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path

from bricknet import parse_sample, score_text


MESHES = None


def init_worker(has_mesh):
    global MESHES
    MESHES = has_mesh


def inspect_row(line):
    row = json.loads(line)
    text = row["prompt"] + row["completion"]
    parsed = parse_sample(text)
    if parsed.error is not None:
        return line, "parse_error"
    if not all(MESHES[part.part_id] for part in parsed.tree.parts):
        return line, "missing_mesh"
    if score_text(text)[2]:
        return line, "collision"
    return line, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--part-names", type=Path, required=True)
    ap.add_argument("--colliders", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    stems = list(json.loads(args.part_names.read_text(encoding="utf-8")))
    has_mesh = [(args.colliders / f"{stem}.cvx.obj").is_file() for stem in stems]
    counts = Counter()
    sources = defaultdict(set)
    splits = ("train", "dev", "test", "val_regression")
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(has_mesh,))
    for split in splits:
        path = args.data / f"{split}.jsonl"
        if not path.exists():
            continue
        temp = path.with_name(path.name + ".screened.tmp")
        with path.open(encoding="utf-8") as inp, temp.open("w", encoding="utf-8") as out:
            for line, reason in pool.map(inspect_row, inp, chunksize=32):
                row = json.loads(line)
                if reason:
                    counts[f"removed_{split}_{reason}"] += 1
                    continue
                out.write(line)
                counts[split] += 1
                counts[f"{split}:{row['condition']}"] += 1
                sources[split].add(row["source"])
                if counts[split] % 10000 == 0:
                    print(f"{split}: {counts[split]} verified", flush=True)
        os.replace(temp, path)
    pool.shutdown()
    metadata_path = args.data / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["counts_before_collider_screen"] = metadata["counts"]
    metadata["counts"] = {k: v for k, v in counts.items() if not k.startswith("removed_")}
    metadata["source_counts"] = {k: len(v) for k, v in sources.items()}
    metadata["collider_screen"] = {k: v for k, v in counts.items() if k.startswith("removed_")}
    metadata["collision_validation"] = "Every retained serialized full path checked by BrickNet score_text with all collider files present; standard 1 LDU tolerance, not mechanical stability."
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["collider_screen"], indent=2))


if __name__ == "__main__":
    main()
