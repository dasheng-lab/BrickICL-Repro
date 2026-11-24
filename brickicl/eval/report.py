"""Summarize frozen-model results: per-condition means and paired differences against a reference condition
(evidence when present, e.g. evidence - ablated is the ICL gain; otherwise the first condition in the file).

Confidence intervals come from a bootstrap over source models (pairs from one model move together).

Usage: python -m brickicl.eval.report pilot.0.6B.jsonl [pilot.1.7B.jsonl ...]
"""

import json
import sys

import numpy as np

METRICS = ("exact", "place_acc", "part_acc", "first_ok", "parse_ok", "lp_target")


def cluster_ci(values, groups, reps=2000, seed=0):
    """95% bootstrap CI of the mean, resampling whole groups."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    keys = np.unique(groups)
    members = [values[groups == g] for g in keys]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        pick = rng.integers(len(keys), size=len(keys))
        means.append(np.concatenate([members[i] for i in pick]).mean())
    return np.percentile(means, [2.5, 97.5])


def report(path):
    rows = [json.loads(line) for line in open(path)]
    head = rows[0]
    print(f"\n## {path}  ({head['model']} + {head['lora']}, {len(rows)} rows, n={head['n']} samples/row)")
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    ref = "evidence" if "evidence" in conds else conds[0]
    others = [c for c in conds if c != ref]
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], {})[r["condition"]] = r
    pairs = [p for p in pairs.values() if set(conds) <= p.keys()]
    metrics = [m for m in METRICS if m == "lp_target" or head["n"]] + (["anchor_lag"] if "anchor_lag" in head else [])
    print(f"{'metric':<10} " + " ".join(f"{c:>13}" for c in [ref] + others) + "   " +
          "   ".join(f"{c} - {ref} [95% CI]" for c in others))
    for m in metrics:
        v = {c: np.array([p[c][m] for p in pairs]) for c in conds}
        src = [p[ref]["source"] for p in pairs]
        diffs = []
        for c in others:
            lo, hi = cluster_ci(v[c] - v[ref], src)
            diffs.append(f"{(v[c] - v[ref]).mean():+8.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"{m:<10} " + " ".join(f"{v[c].mean():13.3f}" for c in [ref] + others) + "   " + "   ".join(diffs))
    print(f"pairs: {len(pairs)} from {len({p[ref]['source'] for p in pairs})} models")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        report(path)
