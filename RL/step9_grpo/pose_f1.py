"""Pose F1 of saved dev samples, two ways, per checkpoint and (stretched dev) per condition. (CPU, bricknet env)

  ours     2 m / (n_p + n_t) over everything written, 0 when the text does not parse (the r1 reward; F1 reward)
  sunshq   Pose F1 on the first n_t parsed parts only (the text cut to n_t parts, then scored), as in sunshq's
           structural_eval table: writing too much is not penalised, stopping short is

Reads <run>/<dir>/step-<s>.jsonl (rows with prefix-less fields: pair, k, n_prefix, samples' text) and the prompts
they were sampled from (--prompts, after the same --first-part transform), joined on pair.

Usage (from ~/RL, `source env.sh`):
  python step9_grpo/pose_f1.py --run runs/v2.1/train/grpo --dir dev_stretch --prompts runs/stretch/dev.jsonl --first-part
"""

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grpo  # noqa: E402
from common import rewards  # noqa: E402
from common.files import read_jsonl  # noqa: E402


def score(job):
    p, texts = job
    ref = rewards.realize(p["prefix"] + p["target"])
    out = []
    for t in texts:
        cut = "".join(t.splitlines(keepends=True)[: 2 * p["k"]])  # every completion part is two lines
        s = rewards.completion(p["prefix"], p["target"], cut, p["n_prefix"], p["k"], ref)
        out.append(s["f1"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--first-part", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    prompts = read_jsonl(args.prompts)
    if args.first_part:
        prompts = grpo.give_first_part_pairs(prompts) if "condition" in prompts[0] else grpo.first_part_all(prompts)
    by_pair = {p["pair"]: p for p in prompts}
    files = sorted(Path(args.run, args.dir).glob("step-*.jsonl"), key=lambda f: int(f.stem.split("-")[1]))
    conds = ("stretched", "original") if "condition" in prompts[0] else (None,)
    head = "| step | " + " | ".join(f"ours{', ' + c if c else ''} | sunshq{', ' + c if c else ''}" for c in conds) + " |"
    print(head + "\n|---:|" + "---:|---:|" * len(conds))
    with get_context("spawn").Pool(args.workers) as pool:
        for f in files:
            rows = read_jsonl(f)
            cut = pool.map(score, [(by_pair[r["pair"]], [x["text"] for x in r["samples"]]) for r in rows], chunksize=4)
            cells = []
            for c in conds:
                sel = [(r, cf) for r, cf in zip(rows, cut) if c is None or r["condition"] == c]
                ours = np.mean([x["f1"] if x["parses"] else 0.0 for r, _ in sel for x in r["samples"]])
                suns = np.mean([v for _, cf in sel for v in cf])
                cells += [f"{ours:.3f}", f"{suns:.3f}"]
            print(f"| {f.stem.split('-')[1]} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
