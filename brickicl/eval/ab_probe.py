"""Two-answer probes on a frozen model (rows from brickicl.sampling.build_probes), likelihood only.

For every row: log p(target_ctx + EOS | prefix) - log p(target_prior + EOS | prefix), i.e. how much the model
prefers the completion that follows the edited context over the one the prior suggests. The report lists,
per probe and group, each variant's mean log-odds, the share of rows preferring the context answer, and the
paired shift against the reference variant (95% CI by bootstrap over source models).

With --n, also n free samples per row (the model stops at its own EOS). Placement is not required: each
sample counts the generated parts that occur only in the context answer (e.g. lime wheels, the decoration)
and those that occur only in the prior answer (wheels in their original colour), matched by part and colour.

Usage:
  python -m brickicl.eval.ab_probe --model Qwen/Qwen3-0.6B --lora kulits/BrickNet-0.6B-PT --data probes.jsonl --out out.jsonl [--n 4]
  python -m brickicl.eval.ab_probe --report out.jsonl [more.jsonl ...]
"""

import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from .frozen import load, sample, target_logprobs
from .report import cluster_ci

KEEP = ("source", "pair", "probe", "group", "variant", "ref_variant", "copies", "n_prefix", "k_prior", "k_ctx",
        "color", "r", "k_stretched", "decoration")
SAMPLE_METRICS = ("hits_ctx", "hits_prior", "any_ctx", "any_prior")


def node_texts(text):
    """'part | colour' of every node line in a completion (node lines are the even lines)."""
    return [line.split(" ", 1)[1] for line in text.splitlines()[0::2] if " " in line]


def sample_stats(row, completions):
    ctx, prior = Counter(node_texts(row["target_ctx"])), Counter(node_texts(row["target_prior"]))
    only_ctx, only_prior = set(ctx - prior), set(prior - ctx)
    hits = [(sum(t in only_ctx for t in node_texts(c)), sum(t in only_prior for t in node_texts(c))) for c in completions]
    m = len(hits) or 1
    return {
        "hits_ctx": sum(h[0] for h in hits) / m,  # mean count per sample of parts only the context answer has
        "hits_prior": sum(h[1] for h in hits) / m,
        "any_ctx": sum(h[0] > 0 for h in hits) / m,  # share of samples with at least one such part
        "any_prior": sum(h[1] > 0 for h in hits) / m,
        "samples": completions,
    }


def run(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [json.loads(line) for line in open(args.data)][: args.limit]
    t0 = time.time()
    model, tok = load(args.model, args.lora, args.dtype, device)
    lp = {}
    for kind in ("prior", "ctx"):
        lp[kind], bad = target_logprobs(
            model, tok, [{"prefix": r["prefix"], "target": r[f"target_{kind}"]} for r in rows], args.score_batch, device
        )
        print(f"{len(rows)} rows, {kind} answers scored ({time.time() - t0:.0f}s); token-boundary mismatches: {bad}", flush=True)
    texts = {}
    if args.n:  # sampling is the slow part: optionally only the first walk seed of every held-out copy
        idx = [i for i, r in enumerate(rows) if not args.sample_seed0 or r["pair"].endswith("-0")]
        sub = sample(model, tok, [rows[i] for i in idx], args.n, args.token_budget, args.seed, device, k_key="k_ctx", extra_parts=4)
        texts = dict(zip(idx, sub))
        print(f"sampled {len(idx)} rows ({time.time() - t0:.0f}s)", flush=True)
    with open(args.out, "w") as f:
        for i, (r, a, b) in enumerate(zip(rows, lp["prior"], lp["ctx"])):
            lo_p, lo_c = a["lp_target"] + a["lp_eos"], b["lp_target"] + b["lp_eos"]
            out = {k: r[k] for k in KEEP if k in r} | {
                "lp_prior": lo_p, "lp_ctx": lo_c, "log_odds": lo_c - lo_p, "model": args.model, "lora": args.lora,
            }
            if i in texts:
                out |= sample_stats(r, texts[i])
            f.write(json.dumps(out) + "\n")
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


def report(path):
    rows = [json.loads(line) for line in open(path)]
    print(f"\n## {path}  ({rows[0]['model']} + {rows[0]['lora']}, {len(rows)} rows)")
    sections = {}
    for r in rows:
        sections.setdefault((r["probe"], r["group"]), []).append(r)
    for (probe, group), rs in sections.items():
        ref = rs[0]["ref_variant"]
        by_pair = {}
        for r in rs:
            by_pair.setdefault(r["pair"], {})[r["variant"]] = r
        variants = list(dict.fromkeys(r["variant"] for r in rs))
        n_models = len({r["source"] for r in rs})
        extra = f", colour {rs[0]['color']}" if probe == "color" else f", r = {rs[0].get('r')}"
        print(f"\n{probe}{' / ' + group if group else ''}  ({len(by_pair)} held-out copies x seeds from {n_models} models{extra})")
        print(f"  {'variant':<11} {'log-odds':>9} {'P(ctx)':>7}   shift vs {ref:<9} 95% CI (by model)")
        for v in variants:
            pairs = [p for p in by_pair.values() if v in p and ref in p]
            lo = np.array([p[v]["log_odds"] for p in pairs])
            line = f"  {v:<11} {lo.mean():9.2f} {np.mean(lo > 0):7.0%}"
            if v != ref:
                d = lo - np.array([p[ref]["log_odds"] for p in pairs])
                lo_ci, hi_ci = cluster_ci(d, [p[v]["source"] for p in pairs])
                line += f"   {d.mean():+8.2f}            [{lo_ci:+.2f}, {hi_ci:+.2f}]"
            print(line)
        sampled = [r for r in rs if "hits_ctx" in r]
        if sampled:
            print(f"  free samples ({len(sampled[0]['samples'])} per row): parts found only in the context / prior answer")
            print(f"  {'variant':<11} {'ctx parts':>9} {'prior parts':>11} {'P(any ctx)':>10} {'P(any prior)':>12}")
            for v in variants:
                vs = [r for r in sampled if r["variant"] == v]
                if vs:
                    m = {k: np.mean([r[k] for r in vs]) for k in SAMPLE_METRICS}
                    print(f"  {v:<11} {m['hits_ctx']:9.2f} {m['hits_prior']:11.2f} {m['any_ctx']:10.0%} {m['any_prior']:12.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", nargs="+")
    ap.add_argument("--model")
    ap.add_argument("--lora", action="append", default=None)
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--score_batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=0, help="free samples per row (0: likelihood only)")
    ap.add_argument("--sample_seed0", action="store_true", help="sample only rows of walk seed 0")
    ap.add_argument("--token_budget", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.report:
        for path in args.report:
            report(path)
    else:
        run(args)


if __name__ == "__main__":
    main()
