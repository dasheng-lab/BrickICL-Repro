"""Stretch probe on a frozen model: does it complete the held-out copy long (context) or short (prior)?

Input: rows from brickicl.sampling.build_stretch (prefix, target_short, target_long, condition).
For every row it records
  - teacher-forced log p of each completion followed by EOS, and the log-odds long vs short;
  - n free samples (no length forcing: the model stops at its own EOS; the token cap is 64 tokens per
    part for k_long + 4 parts, so a sample can run past the copy into further parts), each classified
    as the long copy, the short copy (exact, order-invariant), or other, plus how many parts of the
    stacked type it generated.

With --prime P the first P parts of the held-out copy (identical in the short and long copies) are
appended to the prompt, so the model starts the leg at the right place and only decides how far to
build it. Each sample then gets a leg length: stacked-type parts among the first held-out part and
everything that hangs off it, compared with the short and long copies' counts.

Usage:
  python -m brickicl.eval.length_probe --model Qwen/Qwen3-0.6B --lora kulits/BrickNet-0.6B-PT \
      --data stretch.jsonl --out stretch.0.6B.jsonl [--prime 1]
  python -m brickicl.eval.length_probe --report stretch.0.6B.jsonl [stretch.1.7B.jsonl ...]
"""

import argparse
import json
import time
from collections import Counter

import numpy as np
import torch
from bricknet import parse_sample

from .frozen import load, sample, target_logprobs
from .match import realize, score
from .report import cluster_ci


def stacked_part(row):
    """Node line text (part | color) of the part the long copy has extra copies of."""
    short = Counter(line.split(" ", 1)[1] for line in row["target_short"].splitlines()[0::2])
    long = Counter(line.split(" ", 1)[1] for line in row["target_long"].splitlines()[0::2])
    (extra,) = (long - short).keys()
    return extra


def shared_head(row, prime):
    """The first `prime` held-out parts (two lines each); they must be the same in both copies."""
    short, long = row["target_short"].splitlines(True), row["target_long"].splitlines(True)
    head = short[: 2 * prime]
    assert head == long[: 2 * prime], f"{row['pair']}: short and long copies differ within the primed parts"
    return "".join(head)


def leg_stack(text, n_prefix, stack):
    """Stacked-type parts among part n_prefix (the first held-out part) and everything hanging off it."""
    tree = parse_sample(text).tree
    if len(tree.parts) <= n_prefix:
        return 0
    lines = text.splitlines()
    names = [lines[0 if i == 0 else 2 * i - 1].split(" ", 1)[-1] for i in range(len(tree.parts))]
    leg = {n_prefix}
    for i in range(n_prefix + 1, len(tree.parts)):
        if tree.edges[i - 1].parent in leg:
            leg.add(i)
    return sum(names[i] == stack for i in leg)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [json.loads(line) for line in open(args.data)][: args.limit]
    t0 = time.time()
    model, tok = load(args.model, args.lora, args.dtype, device)
    print(f"{len(rows)} rows, model {args.model} + {args.lora} on {device}", flush=True)
    lp = {}
    for kind in ("short", "long"):
        lp[kind], bad = target_logprobs(
            model, tok, [{"prefix": r["prefix"], "target": r[f"target_{kind}"]} for r in rows], args.score_batch, device
        )
        print(f"teacher forcing ({kind}) done ({time.time() - t0:.0f}s); token-boundary mismatches: {bad}", flush=True)
    heads = [shared_head(r, args.prime) if args.prime else "" for r in rows]
    if args.n:
        prompts = [{"prefix": r["prefix"] + h, "cap": r["k_long"] - args.prime} for r, h in zip(rows, heads)]
        texts = sample(model, tok, prompts, args.n, args.token_budget, args.seed, device, k_key="cap", extra_parts=4)
    else:
        texts = [[]] * len(rows)
    print(f"sampling done ({time.time() - t0:.0f}s)", flush=True)

    keep = ("source", "pair", "condition", "r", "copies", "n_prefix", "k_short", "k_long", "seed")
    with open(args.out, "w") as f:
        for r, ls, ll, head, comps in zip(rows, lp["short"], lp["long"], heads, texts):
            ref = {kind: realize(r["prefix"] + r[f"target_{kind}"]) for kind in ("short", "long")}
            stack = stacked_part(r)
            leg_short = leg_stack(r["prefix"] + r["target_short"], r["n_prefix"], stack)
            leg_long = leg_stack(r["prefix"] + r["target_long"], r["n_prefix"], stack)
            samples = []
            for c in comps:
                c = head + c  # the completion as the model's continuation of the prefix (primed parts included)
                s_short = score(r["prefix"], r["target_short"], c, r["n_prefix"], ref=ref["short"])
                s_long = score(r["prefix"], r["target_long"], c, r["n_prefix"], ref=ref["long"])
                lines = c.splitlines()
                samples.append({
                    "text": c,
                    "category": "long" if s_long.exact else "short" if s_short.exact else "other",
                    "n_parts": s_short.n_generated,
                    "n_stack": sum(line.split(" ", 1)[-1] == stack for line in lines[0::2]),
                    "leg": leg_stack(r["prefix"] + c, r["n_prefix"], stack),
                    "ended": c.endswith("\n") and s_short.parse_error is None,
                })
            m = len(samples) or 1
            cats = Counter(s["category"] for s in samples)
            lo_s, lo_l = ls["lp_target"] + ls["lp_eos"], ll["lp_target"] + ll["lp_eos"]
            out = {key: r[key] for key in keep} | {
                "prime": args.prime,
                "lp_short": lo_s,
                "lp_long": lo_l,
                "log_odds": lo_l - lo_s,  # > 0: the model prefers the long copy
                "frac_long": cats["long"] / m,
                "frac_short": cats["short"] / m,
                "mean_stack": sum(s["n_stack"] for s in samples) / m,
                "mean_parts": sum(s["n_parts"] for s in samples) / m,
                "leg_short": leg_short,
                "leg_long": leg_long,
                "leg_extra": sum(s["leg"] - leg_short for s in samples) / m,  # long copy: + r
                "leg_is_short": sum(s["leg"] == leg_short for s in samples) / m,
                "leg_is_long": sum(s["leg"] == leg_long for s in samples) / m,
                "leg_ge_long": sum(s["leg"] >= leg_long for s in samples) / m,
                "samples": samples,
                "model": args.model,
                "lora": args.lora,
            }
            f.write(json.dumps(out) + "\n")
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


def report(path):
    rows = [json.loads(line) for line in open(path)]
    head = rows[0]
    print(f"\n## {path}  ({head['model']} + {head['lora']}, r = {head['r']}, {len(rows)} rows)")
    sampled = any(r["samples"] for r in rows)
    metrics = ("log_odds",)
    if sampled:
        metrics += ("frac_long", "frac_short", "mean_stack", "mean_parts")
    if sampled and head.get("prime"):
        print(f"(primed with the first {head['prime']} held-out part(s); leg_* = stacked parts on the started leg)")
        metrics += ("leg_extra", "leg_is_short", "leg_is_long", "leg_ge_long")
    by = {}
    for r in rows:
        by.setdefault(r["pair"], {})[r["condition"]] = r
    pairs = [p for p in by.values() if len(p) == 3]
    print(f"{'metric':<11} {'original':>9} {'stretched':>10} {'ablated':>9}   stretched-original  95% CI (by model)")
    for m in metrics:
        v = {c: np.array([p[c][m] for p in pairs]) for c in ("original", "stretched", "ablated")}
        d = v["stretched"] - v["original"]
        lo, hi = cluster_ci(d, [p["original"]["source"] for p in pairs])
        print(f"{m:<11} {v['original'].mean():9.3f} {v['stretched'].mean():10.3f} {v['ablated'].mean():9.3f}   "
              f"{d.mean():+8.3f}  [{lo:+.3f}, {hi:+.3f}]")
    stack_extra = np.mean([p["original"]["k_long"] - p["original"]["k_short"] for p in pairs])
    print(f"triples: {len(pairs)} from {len({p['original']['source'] for p in pairs})} models; "
          f"the long copy has {stack_extra:.0f} more stacked parts than the short one")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", nargs="+", help="summarize result files instead of running")
    ap.add_argument("--model")
    ap.add_argument("--lora", action="append", default=None)
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--token_budget", type=int, default=400_000)
    ap.add_argument("--score_batch", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prime", type=int, default=0, help="held-out parts given in the prompt (0: none)")
    args = ap.parse_args()
    if args.report:
        for path in args.report:
            report(path)
    else:
        run(args)


if __name__ == "__main__":
    main()
