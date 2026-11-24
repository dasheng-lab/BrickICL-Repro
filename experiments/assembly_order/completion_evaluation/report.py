"""Compare models on the completion test: means per model and paired differences, as markdown tables.

All files must come from eval_completion.py on the same test set. Rows are paired across models;
confidence intervals are a bootstrap over source models (rows from one model move together).

Usage (order: baseline first):
  python experiments/assembly_order/completion_evaluation/report.py runs/assembly_order/eval_v2/{frozen,plain,masked}.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from brickicl.eval.report import cluster_ci  # noqa: E402

SAMPLED = ("exact", "place_acc", "first_ok", "part_acc", "anchor_ref", "on_path", "parse_ok")
LABEL = {
    "exact": "exact: all of M, every part placed right",
    "place_acc": "parts of M placed right",
    "first_ok": "first part placed right",
    "part_acc": "right part and colour, any place",
    "anchor_ref": "first part attaches to the reference anchor",
    "on_path": "first part attaches on the current path",
    "parse_ok": "completion parses",
    "nll_M": "loss on M, nats/token",
    "eos_loss": "loss of EOS after M",
}
SUBSETS = (
    ("all rows", lambda r: True),
    ("off-path attachment (jump)", lambda r: r["jump"]),
    ("anchor on the current path (no jump)", lambda r: not r["jump"]),
    ("|M| = 1", lambda r: r["k"] == 1),
    ("|M| 2-4", lambda r: 2 <= r["k"] <= 4),
    ("|M| 5-10", lambda r: 5 <= r["k"] <= 10),
    ("|M| 11+", lambda r: r["k"] >= 11),
)


def value(row, metric):
    if metric == "nll_M":
        return -row["lp_target"] / row["n_target_tokens"]
    if metric == "eos_loss":
        return -row["lp_eos"]
    return row[metric]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    runs = {}
    for path in args.results:
        rows = [json.loads(line) for line in open(path)]
        runs[rows[0]["name"]] = {r["pair"]: r for r in rows}
        print(f"- `{rows[0]['name']}`: {rows[0]['model']} + {rows[0]['lora']}, {len(rows)} rows, n = {rows[0]['n']} samples per row")
    names = list(runs)
    pairs = sorted(set.intersection(*(set(v) for v in runs.values())))
    assert all(len(v) == len(pairs) for v in runs.values()), "the files cover different rows"
    base = runs[names[0]]
    metrics = [m for m in SAMPLED if base[pairs[0]]["n"]] + ["nll_M", "eos_loss"]
    diffs = [(b, a) for i, a in enumerate(names) for b in names[i + 1 :]]  # later minus earlier

    for title, keep in SUBSETS:
        sub = [p for p in pairs if keep(base[p])]
        if not sub:
            continue
        src = [base[p]["source"] for p in sub]
        print(f"\n### {title}: {len(sub)} rows, {len(set(src))} models\n")
        print("| metric | " + " | ".join(names) + " | " + " | ".join(f"{b} − {a} [95% CI]" for b, a in diffs) + " |")
        print("|---|" + "---:|" * (len(names) + len(diffs)))
        for m in metrics:
            v = {n: np.array([value(runs[n][p], m) for p in sub], float) for n in names}
            cells = [f"{v[n].mean():.3f}" for n in names]
            for b, a in diffs:
                d = v[b] - v[a]
                lo, hi = cluster_ci(d, src, reps=args.reps)
                cells.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"| {LABEL[m]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
