"""Measure valid brick prefix length for independent generations."""
import argparse
import json
from pathlib import Path
from statistics import mean

from bricknet import score_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for sample in record["samples"]:
            text = record["prefix"] + sample
            text = "\n".join(text.rstrip("\n").split("\n")[:199])
            actions, invalid, _ = score_text(text)
            rows.append({"id": record["id"], "valid_length": actions if invalid is None else invalid})
    result = {"mean_valid_length": mean(row["valid_length"] for row in rows), "samples": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Mean valid prefix: {result['mean_valid_length']:.2f} bricks ({len(rows)} samples)")


if __name__ == "__main__":
    main()
