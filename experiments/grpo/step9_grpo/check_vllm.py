"""Does vLLM sample the start model like step 3's HF sampler? And how fast? (GPU, vllm env; ask before starting)

Takes the first --rows pilot prompts with |M| <= --max-k, samples them again with vLLM (merged start model, no
adapter, the same protocol), scores them, and prints the step 3 statistics of both side by side, with a bootstrap
interval over prompts for the difference of the mean reward. Writes <out>/check_vllm.json.

Usage (from ~/RL, after `ENV_NAME=vllm source env.sh`):
  python step9_grpo/check_vllm.py --start runs/start_merged/masked --pilot runs/pilot/completion/shard_000.jsonl \\
      --out runs/grpo_check
"""

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grpo import Sampler, group_stats, score_group  # noqa: E402

from common.files import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--extra-parts", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    pool = get_context("spawn").Pool(args.workers)
    tok = AutoTokenizer.from_pretrained(args.start)
    rows = [r for r in read_jsonl(args.pilot) if r["k"] <= args.max_k][: args.rows]
    for r in rows:
        r["prompt_ids"] = tok.encode(r["prefix"], add_special_tokens=False)
    t0 = time.time()
    sampler = Sampler(args, tok)
    t1 = time.time()
    comps = sampler.sample(rows, args.n, args.seed)
    t2 = time.time()
    scored = pool.map(score_group, [(r, [(x[0], x[1]) for x in c]) for r, c in zip(rows, comps)], chunksize=2)
    pool.close()

    hf = group_stats([(r, r["samples"]) for r in rows])
    vl = group_stats(list(zip(rows, scored)))
    a = np.array([np.mean([x["reward"] for x in s]) for s in scored])
    b = np.array([np.mean([x["reward"] for x in r["samples"]]) for r in rows])
    rng = np.random.default_rng(0)
    boot = [(a - b)[rng.integers(len(a), size=len(a))].mean() for _ in range(2000)]
    n_samples = len(rows) * args.n
    new_tokens = sum(len(x[2]) for c in comps for x in c)
    result = {"prompts": len(rows), "samples": n_samples, "hf_pilot": hf, "vllm": vl,
              "reward_diff": float((a - b).mean()), "reward_diff_95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
              "vllm_load_s": t1 - t0, "vllm_sample_s": t2 - t1, "vllm_samples_per_s": n_samples / (t2 - t1),
              "vllm_new_tokens_per_s": new_tokens / (t2 - t1), "mean_new_tokens": new_tokens / n_samples}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "check_vllm.json").write_text(json.dumps(result, indent=1) + "\n")
    print("| statistic | HF (pilot) | vLLM |\n|---|---:|---:|")
    for k in hf:
        print(f"| {k} | {hf[k]:.3f} | {vl[k]:.3f} |")
    print(f"\nmean reward vLLM - HF: {result['reward_diff']:+.4f}, 95% [{result['reward_diff_95'][0]:+.4f}, "
          f"{result['reward_diff_95'][1]:+.4f}]")
    print(f"vLLM: {result['vllm_samples_per_s']:.1f} samples/s, {result['vllm_new_tokens_per_s']:.0f} new tokens/s "
          f"(HF pilot: 0.94 samples/s)")


if __name__ == "__main__":
    main()
