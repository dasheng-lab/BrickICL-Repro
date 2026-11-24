"""Fold the start model's adapters into one plain checkpoint, for vLLM and for the GRPO trainer. (CPU, bricknet env)

vLLM serves one LoRA on top of a base model, but the start model is Qwen3-0.6B + the official PT adapter + the
masked adapter, merged in that order. This writes that merge with the same loader every eval script uses
(brickicl.eval.frozen.load: merge_and_unload in order, bfloat16), so a GRPO adapter trained on top of it is
evaluated by step 7 as start adapters + GRPO adapter, which is the same model.

The bricknet env's transformers (5.x) writes a tokenizer_config.json and config.json that the vllm env's
transformers 4.51 cannot read, so the base model's own tokenizer files are copied instead (the merge does not
touch the tokenizer) together with its config.json (5.x moved rope_theta to a key 4.x does not read).

--check compares the logits of the saved checkpoint with the in-memory merge on a few training prompts.

Usage (from ~/RL):
  python step9_grpo/merge_start.py --out runs/start_merged/masked \\
      --lora kulits/BrickNet-0.6B-PT --lora $NZD_RUNS/train_v2/masked/checkpoint-2000 --check
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import sampling  # noqa: E402
from common.files import write_run_config  # noqa: E402


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def portable(base, out):
    """Make <out> loadable by transformers 4.x: the base model's tokenizer files, and torch_dtype in config.json."""
    import shutil

    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(base, allow_patterns=list(TOKENIZER_FILES)))
    for f in TOKENIZER_FILES:
        shutil.copyfile(src / f, out / f)
    (out / "chat_template.jinja").unlink(missing_ok=True)  # the base tokenizer_config.json has its own
    # the merge does not change the architecture, and 5.x moved rope_theta into rope_parameters, which 4.x reads
    # as the default 10000 instead of 1e6: use the base model's config.json
    shutil.copyfile(Path(snapshot_download(base, allow_patterns=["config.json"])) / "config.json", out / "config.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lora", action="append", required=True, help="adapters merged in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--portable-only", action="store_true", help="only fix the files of an existing --out")
    ap.add_argument("--pool", default=None, help="pool for --check prompts (default $NZD_RUNS/sft_train_v2)")
    args = ap.parse_args()

    out = Path(args.out)
    if args.portable_only:
        portable(args.model, out)
        print(f"made {out} readable by transformers 4.x")
        return
    model, tok = sampling.load(args.model, args.lora, torch.bfloat16, "cpu")
    model.save_pretrained(out, safe_serialization=True)
    portable(args.model, out)
    write_run_config(out, args, sampling.EXTERNAL, name="merge_config.json")
    print(f"saved {args.model} + {args.lora} -> {out}")

    if args.check:
        import os

        from transformers import AutoModelForCausalLM

        pool = args.pool or os.path.join(os.environ["NZD_RUNS"], "sft_train_v2")
        rows = sampling.prompts(pool, rows=4, max_k=10, seed=123)
        saved = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).eval()
        worst = 0.0
        with torch.inference_mode():
            for r in rows:
                ids = torch.tensor([tok.encode(r["prefix"] + r["target"], add_special_tokens=False)])
                a, b = model(ids).logits.float(), saved(ids).logits.float()
                worst = max(worst, (a - b).abs().max().item())
        result = {"prompts": len(rows), "max_abs_logit_diff": worst}
        (out / "merge_check.json").write_text(json.dumps(result) + "\n")
        print(json.dumps(result))
        assert worst == 0.0, "the saved checkpoint does not reproduce the merge"


if __name__ == "__main__":
    main()
