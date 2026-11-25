"""Create model-specific data manifests and validate loss masking for training rows."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from brickicl.posttrain.data import CompletionCollator, encode_example


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    args.data = args.data.resolve()
    meta = json.loads((args.data / "metadata.json").read_text())
    max_length = meta["max_tokens_with_eos"]
    train_name = "train.jsonl" if meta.get("version") in (2, 3) else "train_balanced.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    train = [json.loads(s) for s in (args.data / train_name).open(encoding="utf-8")]
    allowed = {hashlib.sha256((r["prompt"] + r["completion"]).encode()).digest()
               for r in map(json.loads, (args.data / "train.jsonl").open(encoding="utf-8"))}
    hashes = set()
    prompt_lengths, target_lengths = [], []
    first = []
    for row in train:
        digest = hashlib.sha256((row["prompt"] + row["completion"]).encode()).digest()
        assert digest in allowed and digest not in hashes
        hashes.add(digest)
        assert row["split"] == "train"
        encoded = encode_example(row, tokenizer, max_length=max_length)
        n_prompt = encoded["labels"].count(-100)
        prompt_lengths.append(n_prompt)
        target_lengths.append(len(encoded["input_ids"]) - n_prompt)
        assert len(encoded["input_ids"]) == row["tokens_with_eos"]
        assert encoded["labels"][-1] == tokenizer.eos_token_id
        if len(first) < 4:
            first.append(encoded)
    batch = CompletionCollator(tokenizer)(first)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    for i, encoded in enumerate(first):
        assert batch["labels"][i, len(encoded["labels"]) - 1].item() == tokenizer.eos_token_id
    profile = {
        "rows": len(train), "train_file": train_name,
        "sources": len({(r['source_split'], r["source"]) for r in train}),
        "conditions": dict(Counter(r["condition"] for r in train)),
        "tokens_per_epoch_including_context": sum(prompt_lengths) + sum(target_lengths),
        "supervised_tokens_per_epoch_including_eos": sum(target_lengths),
        "prompt_tokens_p50_p95_max": np.quantile(prompt_lengths, [.5, .95, 1]).tolist(),
        "target_tokens_p50_p95_max": np.quantile(target_lengths, [.5, .95, 1]).tolist(),
        "max_sequence_tokens": max_length,
        "target_parts_min_p50_p95_max": np.quantile([r['k'] for r in train], [0,.5,.95,1]).tolist(),
        "supervised_fraction_of_all_tokens": sum(target_lengths)/(sum(prompt_lengths)+sum(target_lengths)),
        "loss_mask_boundary_eos_padding_checks": "passed for every training row and a padded batch",
    }
    (args.data / "training_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    for size in ("0.6B", "1.7B", "4B"):
        config = {
            "schema": "brickicl-data-manifest-v1",
            "base_model": f"Qwen/Qwen3-{size}", "initial_pt_adapter": f"kulits/BrickNet-{size}-PT",
            "train_file": str(args.data / train_name), "shuffle_training_rows": True,
            "dev_file": str(args.data / "dev.jsonl"), "test_file": str(args.data / "test.jsonl"),
            "regression_file": str(args.data / "val_regression.jsonl") if (args.data / "val_regression.jsonl").exists() else None,
            "format": "prompt_completion", "prompt_field": "prompt", "completion_field": "completion",
            "apply_chat_template": False, "loss_on_prompt": False, "append_eos": True,
            "max_length": max_length, "packing": False,
            "encoding_function": "brickicl.posttrain.data.encode_example",
            "collator": "brickicl.posttrain.data.CompletionCollator",
            "notes": "Data manifest, not a trainer configuration. Merge frozen PT adapter before creating a fresh trainable LoRA; choose checkpoints with dev only.",
        }
        (args.data / f"data.{size}.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
