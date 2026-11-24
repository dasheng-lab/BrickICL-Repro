"""The stretched-reference dev check for checkpoints of a finished run (r1 had no --stretch-dev). (GPU, vllm env)

Same sampling and scoring as grpo.py's in-training check (Sampler, score_group, stretch_stats; --dev-n samples,
seed + 7), so the numbers line up with runs that had --stretch-dev. Writes <run>/dev_stretch/step-<s>.jsonl and
<run>/dev_stretch.jsonl, and prints per checkpoint: long-feature hits per condition, sunshq's Δlog-odds with a 95%
interval (2000 resamples of the pairs, seed 0), stopped-before-target and empty rates, F1 reward.

Rewards are recorded under --beta / --fail-reward as given (default r1's: F1, 0), F1 always.
--first-part gives M's first part in the prompt (sunshq's p1, grpo.py's --first-part), writing dev_stretch_p1/;
--dev-test also samples the ordinary dev test the same way (dev_p1/ or dev_free/), to compare with a run's dev/.

Usage (from ~/RL, `ENV_NAME=vllm source env.sh`):
  python step9_grpo/eval_stretch_dev.py --run runs/r1/train/grpo --steps 0 50 100 150 200 250 300 350 400
"""

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grpo  # noqa: E402
from common.files import read_jsonl, write_jsonl  # noqa: E402


def interval(groups, seed=0, n=2000):
    """95% interval of the mean Δlog-odds over pairs, resampling pairs."""
    by = {}
    for r, g in groups:
        by.setdefault((r["source"], r["sample_index"]), {})[r["condition"]] = g
    d = np.array([grpo.logodds(sum(x["long_hit"] for x in p["stretched"]), len(p["stretched"]))
                  - grpo.logodds(sum(x["long_hit"] for x in p["original"]), len(p["original"]))
                  for p in by.values() if len(p) == 2])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a grpo.py output folder with checkpoint-<step>/")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--stretch-dev", default="runs/stretch/dev.jsonl")
    ap.add_argument("--start", default="runs/start_merged/masked")
    ap.add_argument("--dev-n", type=int, default=4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--fail-reward", type=float, default=0.0)
    ap.add_argument("--extra-parts", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first-part", action="store_true")
    ap.add_argument("--dev-test", help="also sample this dev test (e.g. runs/dev_test/test.jsonl)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    run = Path(args.run)
    pool = get_context("spawn").Pool(args.workers)  # before vLLM and CUDA
    tok = AutoTokenizer.from_pretrained(args.start)
    rows = read_jsonl(args.stretch_dev)
    dev = read_jsonl(args.dev_test) if args.dev_test else []
    if args.first_part:
        rows, dev = grpo.give_first_part_pairs(rows), grpo.first_part_all(dev)
    tag = "_p1" if args.first_part else ""
    for r in rows + dev:
        r["prompt_ids"] = tok.encode(r["prefix"], add_special_tokens=False)
    print(f"{len(rows)} stretched dev rows, {len(dev)} dev test rows, first part given: {args.first_part}")
    sampler = grpo.Sampler(SimpleNamespace(**vars(args)), tok)
    out, out_dev = [], []
    print("| step | hit, stretched | hit, original | Δlog-odds [95%] | short, stretched | empty | F1 reward |")
    print("|---:|---:|---:|---|---:|---:|---:|")
    for i, step in enumerate(args.steps):
        sampler.set_adapter(run / f"checkpoint-{step}", i + 1)
        comps = sampler.sample(rows, args.dev_n, args.seed + 7)
        scored = pool.map(grpo.score_group, grpo.jobs(rows, comps, args), chunksize=4)
        write_jsonl(run / f"dev_stretch{tag}" / f"step-{step}.jsonl",
                    [{k: r[k] for k in grpo.KEEP if k in r} | {"samples": s} for r, s in zip(rows, scored)])
        groups = list(zip(rows, scored))
        st = grpo.group_stats(groups) | grpo.stretch_stats(groups)
        lo, hi = interval(groups)
        short_st = grpo.group_stats([g for g in groups if g[0]["condition"] == "stretched"])["short"]
        st |= {"step": step, "dlogodds_lo": lo, "dlogodds_hi": hi, "short, stretched": short_st}
        out.append(st)
        print(f"| {step} | {st['hit, stretched']:.3f} | {st['hit, original']:.3f} | {st['dlogodds']:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}] | {short_st:.3f} | {st['empty']:.3f} | {st['reward_f1']:.4f} |", flush=True)
        if dev:
            comps = sampler.sample(dev, args.dev_n, args.seed + 7)
            sd = pool.map(grpo.score_group, grpo.jobs(dev, comps, args), chunksize=4)
            write_jsonl(run / f"dev{tag or '_free'}" / f"step-{step}.jsonl",
                        [{k: r[k] for k in grpo.KEEP if k in r} | {"samples": s} for r, s in zip(dev, sd)])
            d = grpo.group_stats(list(zip(dev, sd))) | {"step": step}
            out_dev.append(d)
            print(f"  dev test, step {step}: F1 reward {d['reward_f1']:.4f} reward {d['reward']:.4f} "
                  f"empty {d['empty']:.3f} short {d['short']:.3f} n_p/n_t {d['np_over_nt']:.2f} "
                  f"exact {d['exact']:.4f}", flush=True)
    write_jsonl(run / f"dev_stretch{tag}.jsonl", out)
    if out_dev:
        write_jsonl(run / f"dev{tag or '_free'}.jsonl", out_dev)
    pool.close()


if __name__ == "__main__":
    main()
