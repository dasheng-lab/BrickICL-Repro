"""Training adapter for plain path completion; no chat template or internal EOS."""
from dataclasses import dataclass


def encode_example(row, tokenizer, max_length=2816):
    prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
    answer = tokenizer.encode(row["completion"], add_special_tokens=False)
    joint = tokenizer.encode(row["prompt"] + row["completion"], add_special_tokens=False)
    if joint != prefix + answer:
        raise ValueError("token boundary changed at prompt/completion split")
    ids = joint + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        raise ValueError("example exceeds max_length; do not silently truncate")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + answer + [tokenizer.eos_token_id]}


@dataclass
class CompletionCollator:
    tokenizer: object

    def __call__(self, features):
        import torch
        size = max(len(x["input_ids"]) for x in features)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            n = size - len(f["input_ids"])
            result["input_ids"].append(f["input_ids"] + [pad] * n)
            result["attention_mask"].append(f["attention_mask"] + [0] * n)
            result["labels"].append(f["labels"] + [-100] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
