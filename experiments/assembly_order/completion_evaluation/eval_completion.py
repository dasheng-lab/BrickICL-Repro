"""Completion test: give a model the prefix G \\ M and let it write M.

For every row of a test set from make_test.py it records

  - the teacher-forced log-likelihood of the reference M, and of EOS right after it;
  - n sampled completions of exactly |M| parts, each scored order-invariantly against M
    (part and colour, pose, exact), as in brickicl.eval;
  - where the first generated part attaches: to the reference anchor, and whether to the last
    prefix part or one of its ancestors (the only places a depth-first sequence would continue).

Model loading, sampling (temperature 1.0, top-k 20, top-p 0.95) and the stop rule (EOS banned until
2|M| new lines, then forced) are brickicl.eval.frozen's, which follow upstream scripts/generate.py.
Adapters given with --lora are merged in order: the official PT first, then one of our checkpoints.

Usage (GPU machine, from the repo root):
  python experiments/assembly_order/completion_evaluation/eval_completion.py --name masked --data runs/assembly_order/eval_v2/test.jsonl \\
      --lora kulits/BrickNet-0.6B-PT --lora runs/assembly_order/train_v2/masked/checkpoint-2000 \\
      --out runs/assembly_order/eval_v2/masked.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from bricknet import parse_sample  # noqa: E402

from brickicl.eval.frozen import load, sample, target_logprobs, upstream_stop_after_newlines  # noqa: E402
from brickicl.eval.match import realize, score  # noqa: E402


def parents(text):
    """Parent position of every part in the longest parsable head of a path text (None for the root)."""
    return [None] + [e.parent for e in parse_sample(text).tree.edges]


def on_path(par, last, anchor):
    """Is `anchor` the part `last` or one of its ancestors?"""
    u = last
    while u is not None and u != anchor:
        u = par[u]
    return u is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="label of this model in the report (frozen, plain, masked)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lora", action="append", default=None, help="adapters merged in order")
    ap.add_argument("--n", type=int, default=8, help="samples per row (0: likelihood only)")
    ap.add_argument("--token-budget", type=int, default=250_000, help="max sequences x tokens per generate call")
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke tests)")
    args = ap.parse_args()
    adapters = args.lora or ["kulits/BrickNet-0.6B-PT"]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [json.loads(line) for line in open(args.data)][: args.limit]
    t0 = time.time()
    model, tok = load(args.model, adapters, args.dtype, device)
    print(f"[{args.name}] {len(rows)} rows, {args.model} + {adapters} on {device}", flush=True)

    lps, mismatched = target_logprobs(model, tok, rows, args.score_batch, device)
    print(f"[{args.name}] teacher forcing done ({time.time() - t0:.0f} s); token-boundary mismatches: {mismatched}")
    if args.n:
        texts = sample(model, tok, rows, args.n, args.token_budget, args.seed, device, upstream_stop_after_newlines())
    else:
        texts = [[]] * len(rows)
    print(f"[{args.name}] sampling done ({time.time() - t0:.0f} s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keep = ("source", "pair", "nodes", "k", "n_prefix", "jump", "lag")
    with open(args.out, "w") as f:
        for r, lp, comps in zip(rows, lps, texts):
            n0, k = r["n_prefix"], r["k"]
            ref = realize(r["prefix"] + r["target"])
            ref_par = parents(r["prefix"] + r["target"])
            samples = []
            for c in comps:
                s = score(r["prefix"], r["target"], c, n0, ref=ref).asdict()
                par = parents(r["prefix"] + c)
                anchor = par[n0] if len(par) > n0 else None  # where the first generated part attaches
                s["first_anchor"] = anchor
                s["anchor_ref"] = anchor is not None and anchor == ref_par[n0]
                s["on_path"] = anchor is not None and on_path(par, n0 - 1, anchor)
                samples.append({"text": c} | s)
            m = len(samples) or 1
            row = {key: r[key] for key in keep} | lp | {
                "ref_anchor": ref_par[n0],
                "n": len(samples),
                "parse_ok": sum(s["parse_error"] is None for s in samples) / m,
                "part_acc": sum(s["part_hits"] for s in samples) / (m * k),
                "place_acc": sum(s["placed"] for s in samples) / (m * k),
                "first_ok": sum(s["first_ok"] for s in samples) / m,
                "exact": sum(s["exact"] for s in samples) / m,
                "anchor_ref": sum(s["anchor_ref"] for s in samples) / m,
                "on_path": sum(s["on_path"] for s in samples) / m,
                "samples": samples,
                "name": args.name,
                "model": args.model,
                "lora": adapters,
            }
            f.write(json.dumps(row) + "\n")
    print(f"[{args.name}] wrote {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
