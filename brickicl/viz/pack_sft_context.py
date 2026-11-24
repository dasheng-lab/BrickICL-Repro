"""Pack BrickNet-SFT caption + partial-path results for the web viewer."""
import argparse
import json
from pathlib import Path

from bricknet import parse_sample

from brickicl.viz.pack_probe import Packer, first_sentence


CONDITION_TAGS = {
    "original": "Original",
    "stretched": "Stretched",
    "ablated": "Ablated",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(row):
    expected_prompt = row["caption"] + "\n" + row["query_prefix"]
    if row["prompt"] != expected_prompt:
        raise ValueError(f"bad SFT prompt: {row['pair']} {row['condition']}")
    parsed = parse_sample(row["query_prefix"])
    if parsed.error is not None:
        raise ValueError(f"bad query path: {row['pair']} {row['condition']}: {parsed.error}")
    if len(parsed.tree.parts) != row["n_prefix"]:
        raise ValueError(f"bad query prefix count: {row['pair']} {row['condition']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--sources", type=int, nargs="*")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    rows = {(row["pair"], row["condition"]): row for row in read_jsonl(args.data)}
    for row in rows.values():
        validate(row)

    packer = Packer()
    references = set()
    for path in args.results:
        for result in read_jsonl(path):
            if args.sources and result["source"] not in args.sources:
                continue
            key = result["pair"], result["condition"]
            row = rows[key]
            source = result["source"]
            condition = result["condition"]
            walk_seed = result.get("walk_seed_index", row.get("walk_seed_index", 0))
            size = result["model"].split("-")[-1]
            tag = CONDITION_TAGS[condition]
            group = f"SFT caption + partial r={result['r']} - walk seed {walk_seed} - #{source}"
            caption = first_sentence(row["caption"])
            stem = f"sft_{source}_w{walk_seed}_{condition}"
            expected = row["target_long" if condition == "stretched" else "target_short"]

            if key not in references:
                packer.add(
                    f"sft_context/{stem}_reference.mpd",
                    row["query_prefix"] + expected,
                    row["n_prefix"],
                    group=group,
                    caption=caption,
                    title=f"{tag} - reference",
                    note="The caption and partial assembly are the SFT prompt; highlighted parts are the reference continuation.",
                )
                references.add(key)

            for sample_index, sample in enumerate(result["samples"][: args.samples], 1):
                packer.add(
                    f"sft_context/{stem}_{size}_sample{sample_index}.mpd",
                    row["query_prefix"] + sample["raw_completion"],
                    row["n_prefix"],
                    group=group,
                    caption=caption,
                    title=f"{tag} - SFT {size} - sample {sample_index}",
                    note=(f"Prompt: validation caption plus partial assembly. Generated leg: {sample['leg']} parts; "
                          f"short reference: {result['leg_short']}; long reference: {result['leg_long']}; "
                          f"parse error: {sample['parse_error'] or 'none'}."),
                )
    packer.save()


if __name__ == "__main__":
    main()
