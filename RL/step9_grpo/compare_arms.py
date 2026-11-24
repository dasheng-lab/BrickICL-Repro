"""Compare GRPO arms checkpoint by checkpoint on the dev test and the stretched dev set. (CPU, any env with numpy)

Each arm is NAME=RUN:DEV_DIR:STRETCH_DIR (folders of step-<s>.jsonl with scored samples, as grpo.py and
eval_stretch_dev.py write them). The first arm is the reference; for every other arm and every step both have,
it prints the arm's value and the difference with a 95% interval from 2000 paired resamples (seed 0): of the
prompts for dev F1 reward, of the held-out copies (pairs) for Δlog-odds. Both arms must have sampled the same
prompts (same dev files, same transform).

Usage (from ~/RL):
  python step9_grpo/compare_arms.py r1=runs/r1/train/grpo:dev_p1:dev_stretch_p1 v2.1=runs/v2.1/train/grpo:dev:dev_stretch
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def f1_reward(x):
    return x.get("reward_f1", x["reward"]) if "reward_f1" in x else (x["f1"] if x["parses"] else 0.0)


def logodds(k, n):
    return math.log((k + 0.5) / (n - k + 0.5))


def dev_metrics(rows):
    s = [(r, x) for r in rows for x in r["samples"]]
    return {
        "per_prompt": {r["pair"]: np.mean([f1_reward(x) for x in r["samples"]]) for r in rows},
        "F1 reward": np.mean([f1_reward(x) for _, x in s]),
        "exact": np.mean([x["exact"] for _, x in s]),
        "empty": np.mean([x["n_generated"] == 0 for _, x in s]),
        "short": np.mean([x["parses"] and x["n_generated"] < r["k"] for r, x in s]),
        "n_p/n_t": np.mean([x["n_generated"] / r["k"] for r, x in s]),
    }


def stretch_metrics(rows):
    by = {}
    for r in rows:
        by.setdefault(r["pair"].rsplit(":", 1)[0], {})[r["condition"]] = r  # "stretch:<source>:<copy>:<condition>"
    d = {k: logodds(sum(x["long_hit"] for x in p["stretched"]["samples"]), len(p["stretched"]["samples"]))
         - logodds(sum(x["long_hit"] for x in p["original"]["samples"]), len(p["original"]["samples"]))
         for k, p in by.items() if len(p) == 2}
    st = [(r, x) for r in rows if r["condition"] == "stretched" for x in r["samples"]]
    og = [x for r in rows if r["condition"] == "original" for x in r["samples"]]
    return {
        "per_pair": d,
        "dlogodds": np.mean(list(d.values())),
        "hit, stretched": np.mean([x["long_hit"] for _, x in st]),
        "hit, original": np.mean([x["long_hit"] for x in og]),
        "short, stretched": np.mean([x["parses"] and x["n_generated"] < r["k"] for r, x in st]),
        "F1 reward, stretched dev": np.mean([f1_reward(x) for r in rows for x in r["samples"]]),
    }


def paired(a, b, seed=0, n=2000):
    keys = sorted(set(a) & set(b))
    d = np.array([b[k] - a[k] for k in keys])
    idx = np.random.default_rng(seed).integers(len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(keys)


def load(spec):
    name, rest = spec.split("=", 1)
    run, dev, stretch = rest.split(":")
    run = Path(run)
    steps = sorted({int(p.stem.split("-")[1]) for p in (run / dev).glob("step-*.jsonl")}
                   & {int(p.stem.split("-")[1]) for p in (run / stretch).glob("step-*.jsonl")})
    return name, {s: dev_metrics(read(run / dev / f"step-{s}.jsonl")) | stretch_metrics(read(run / stretch / f"step-{s}.jsonl"))
                  for s in steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--json", help="also write the table here")
    args = ap.parse_args()
    arms = [load(a) for a in args.arms]
    ref_name, ref = arms[0]
    cols = ["F1 reward", "exact", "empty", "short", "n_p/n_t", "hit, stretched", "hit, original", "short, stretched",
            "dlogodds"]
    out = []
    print("| arm | step | " + " | ".join(cols) + " | Δ F1 reward vs " + ref_name + " [95%] | Δ dlogodds vs "
          + ref_name + " [95%] |")
    print("|---|---:|" + "---:|" * len(cols) + "---|---|")
    for name, m in arms:
        for s, v in m.items():
            row = {"arm": name, "step": s} | {c: float(v[c]) for c in cols}
            extra = ["", ""]
            if name != ref_name and s in ref:
                f = paired(ref[s]["per_prompt"], v["per_prompt"])
                d = paired(ref[s]["per_pair"], v["per_pair"])
                row |= {"dF1": f[:3], "ddlogodds": d[:3], "n_prompts": f[3], "n_pairs": d[3]}
                extra = [f"{f[0]:+.3f} [{f[1]:+.3f}, {f[2]:+.3f}]", f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"]
            out.append(row)
            print(f"| {name} | {s} | " + " | ".join(f"{row[c]:.3f}" if c != "dlogodds" else f"{row[c]:+.2f}"
                                               for c in cols) + f" | {extra[0]} | {extra[1]} |")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
