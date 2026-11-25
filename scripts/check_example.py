"""Validate locally supplied completion examples and optional token boundaries."""
import argparse
from pathlib import Path

from brickicl.jsonl import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "data/smoke_example.jsonl")
    parser.add_argument("--tokenizer", help="Optional tokenizer directory or public model ID")
    args = parser.parse_args()
    try:
        records = read_jsonl(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    from bricknet import parse_sample, score_text

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    for number, record in enumerate(records, 1):
        prefix = record.get("prefix")
        paired_target = record.get("target_short") if record.get("condition") == "original" else (
            record.get("target_long") if record.get("condition") == "stretched" else None
        )
        completion = record.get("target") or record.get("completion") or paired_target
        if not isinstance(prefix, str) or not prefix.strip() or not isinstance(completion, str) or not completion.strip():
            parser.error(f"Example {number} needs a nonempty prefix and a completion, target, or condition-specific paired target")
        assembly = prefix + completion
        parsed = parse_sample(assembly)
        if parsed.error is not None:
            parser.error(f"Example {number} cannot be parsed: {parsed.error}")
        _, invalid, collisions = score_text(assembly)
        if invalid is not None or collisions:
            parser.error(f"Example {number} has invalid connections or collisions")
        if tokenizer is not None:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            if tokenizer.encode(assembly, add_special_tokens=False) != prefix_ids + completion_ids:
                parser.error(f"Example {number} changes tokens across the completion boundary")
    print(f"Validated {len(records)} local completion examples.")


if __name__ == "__main__":
    main()
