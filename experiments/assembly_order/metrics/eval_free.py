"""Completion with the model's own stop: give the prefix G \\ M, let the model write until it emits EOS.

Steps 5 and 8 force every completion to exactly |M| parts. The report's metrics (Pose F1 with the number of
generated parts in the denominator, "nothing generated counts as a failure") need the model to decide when
to stop, as in the teammates' evaluation (brickicl/posttrain/evaluate_posttrain.py). The settings follow it:
temperature 1.0, top-k 20, top-p 0.95, EOS allowed from the first token, at most 64 (|M| + 4) + 32 new tokens.

For every row of a step 5 format test set it writes the teacher-forced log-likelihood of M and of EOS, and n
samples with: the text, whether EOS was emitted, and the quantities of metrics.py.

Usage (GPU machine, from the repo root):
  python experiments/assembly_order/metrics/eval_free.py --name masked \\
      --data runs/assembly_order/eval_shared/longleg/test.jsonl \\
      --lora kulits/BrickNet-0.6B-PT --lora runs/assembly_order/train_v2/masked/checkpoint-2000 \\
      --out runs/assembly_order/report_metrics/completion/longleg/masked.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from metrics import score_completion  # noqa: E402

from brickicl.eval.frozen import _chunks, load, target_logprobs  # noqa: E402
from brickicl.eval.match import realize  # noqa: E402


@torch.inference_mode()
def sample_free(model, tok, rows, n, budget, seed, device, extra_parts):
    """n completions per row as (text, EOS emitted). Rows sharing |M| share a token cap and are batched."""
    eos = tok.eos_token_id
    prompts = [tok.encode(r["prefix"], add_special_tokens=False) for r in rows]
    lengths = [len(p) for p in prompts]
    result = [None] * len(rows)
    by_k = {}
    for i, r in enumerate(rows):
        by_k.setdefault(r["k"], []).append(i)
    for k, idx in sorted(by_k.items()):
        done, cap = 0, 64 * (k + extra_parts) + 32
        for chunk in _chunks(idx, lengths, n, cap, budget):
            L = max(lengths[i] for i in chunk)
            inp = torch.tensor([[tok.pad_token_id] * (L - lengths[i]) + prompts[i] for i in chunk for _ in range(n)],
                               device=device)
            attn = torch.zeros_like(inp)
            for j, i in enumerate(chunk):
                attn[j * n : (j + 1) * n, L - lengths[i] :] = 1
            torch.manual_seed(seed + chunk[0])
            out = model.generate(inp, attention_mask=attn, do_sample=True, temperature=1.0, top_k=20, top_p=0.95,
                                 max_new_tokens=cap, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            new = out[:, L:].tolist()
            for j, i in enumerate(chunk):
                result[i] = []
                for ids in new[j * n : (j + 1) * n]:
                    stopped = eos in ids
                    ids = ids[: ids.index(eos)] if stopped else ids
                    result[i].append((tok.decode(ids, skip_special_tokens=True), stopped))
            done += len(chunk)
            print(f"  sampled k={k}: {done}/{len(idx)} rows", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="label of this model in the report")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lora", action="append", default=None, help="adapters merged in order")
    ap.add_argument("--n", type=int, default=4, help="samples per row")
    ap.add_argument("--extra-parts", type=int, default=4, help="the token cap allows |M| + this many parts")
    ap.add_argument("--token-budget", type=int, default=250_000, help="max sequences x tokens per generate call")
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-model", type=int, default=None, help="at most this many rows per val model, in file order")
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke tests)")
    args = ap.parse_args()
    adapters = args.lora or ["kulits/BrickNet-0.6B-PT"]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    rows, seen = [], {}
    for r in map(json.loads, open(args.data)):
        seen[r["source"]] = seen.get(r["source"], 0) + 1
        if args.per_model is None or seen[r["source"]] <= args.per_model:
            rows.append(r)
    rows = rows[: args.limit]
    t0 = time.time()
    model, tok = load(args.model, adapters, args.dtype, device)
    print(f"[{args.name}] {len(rows)} rows, {args.model} + {adapters} on {device}", flush=True)
    lps, mismatched = target_logprobs(model, tok, rows, args.score_batch, device)
    print(f"[{args.name}] teacher forcing done ({time.time() - t0:.0f} s); token-boundary mismatches: {mismatched}")
    texts = sample_free(model, tok, rows, args.n, args.token_budget, args.seed, device, args.extra_parts)
    print(f"[{args.name}] sampling done ({time.time() - t0:.0f} s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keep = ("source", "pair", "nodes", "k", "n_prefix", "jump", "lag", "kind", "copies")
    with open(args.out, "w") as f:
        for r, lp, comps in zip(rows, lps, texts):
            ref = realize(r["prefix"] + r["target"])
            samples = [{"text": text, "eos": stopped} | score_completion(r["prefix"], r["target"], text, r["n_prefix"], ref)
                       for text, stopped in comps]
            f.write(json.dumps({key: r[key] for key in keep if key in r} | lp | {
                "samples": samples, "name": args.name, "model": args.model, "lora": adapters,
                "protocol": {"stop": "eos", "extra_parts": args.extra_parts, "n": args.n, "seed": args.seed},
            }) + "\n")
    print(f"[{args.name}] wrote {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
