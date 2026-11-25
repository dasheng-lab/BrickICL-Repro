"""Sample LEGO continuations from a base model plus optional LoRA adapters."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from brickicl.jsonl import read_jsonl


class SuppressEOS(LogitsProcessor):
    def __init__(self, eos_id):
        self.eos_id = eos_id

    def __call__(self, input_ids, scores):
        scores[:, self.eos_id] = -float("inf")
        return scores


def target_logp(model, tokenizer, prefix, target):
    """Conditional log probability of the whole target and EOS."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prefix_ids or tokenizer.encode(prefix + target, add_special_tokens=False) != prefix_ids + target_ids[:-1]:
        raise ValueError("The tokenizer changed tokens across the prefix/target boundary")
    ids = torch.tensor([prefix_ids + target_ids], device=model.device)
    with torch.inference_mode():
        logits = model(ids).logits[0, len(prefix_ids) - 1:-1]
        selected = logits.log_softmax(-1).gather(
            1, torch.tensor(target_ids, device=model.device)[:, None]
        )
    return float(selected.sum().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base Hugging Face model ID or local path")
    parser.add_argument("--adapter", action="append", default=[], help="Optional LoRA adapter; repeat in merge order")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id and prefix fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-eos", action="store_true", help="Independent-generation stopping control")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    try:
        rows = read_jsonl(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for number, row in enumerate(rows, 1):
        if not isinstance(row.get("prefix"), str) or not row["prefix"].strip():
            parser.error(f"Input row {number} needs a nonempty prefix string")

    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    for adapter in args.adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processors = [SuppressEOS(tokenizer.eos_token_id)] if args.no_eos else []
    with args.output.open("w", encoding="utf-8") as dest:
        for number, row in enumerate(rows):
            prefix = row["prefix"]
            inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(model.device)
            prediction = []
            for draw in range(args.samples):
                torch.manual_seed(20261124 + 1009 * number + draw)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, do_sample=True, temperature=1.0, top_k=20,
                        top_p=0.95, max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        logits_processor=processors,
                    )
                prediction.append(tokenizer.decode(
                    output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
                ))
            if row.get("condition") == "original":
                target = row.get("target_short")
            elif row.get("condition") == "stretched":
                target = row.get("target_long")
            else:
                target = None
            target = row.get("target") or row.get("completion") or target
            result = {
                "id": row.get("id", f"row_{number}"),
                "source_key": row.get("source_key", row.get("id", f"row_{number}")),
                "condition": row.get("condition"),
                "prefix": prefix,
                "target": target,
                "samples": prediction,
            }
            if row.get("target_long") is not None:
                result["target_logp"] = target_logp(model, tokenizer, prefix, row["target_long"])
            dest.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
