"""First part right, for every dev evaluation of a GRPO run. (CPU, bricknet env)

"First part right" as in nzd's step 5 metric (brickicl.eval.match.Score.first_ok): the first generated part after
the prefix is a correct placement of some held-out part (same part, pose equivalent by the connector test,
3 LDU). It is reported two ways: with colour, exactly nzd's first_ok, and pose only (colour ignored, like our
reward). A sample whose first part does not parse counts as wrong. Also the same for the next-in-path part only
(the first generated part equals the first target part), which is the stricter reading.

Reads <run>/dev/step-*.jsonl (eval_free format: samples' text) and the dev test (prefix, target) joined on pair.
Writes <run>/first_part.json and prints a table.

Usage (from ~/RL):
  python step9_grpo/first_part.py --run runs/r1/train/grpo --dev runs/dev_test/test.jsonl
"""

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.files import read_jsonl  # noqa: E402


def first_part(job):
    """(prefix, target, n_prefix, texts) -> per text (any colour, any pose-only, next colour, next pose-only)."""
    from brickicl.eval.match import realize, same_pose

    prefix, target, n_prefix, texts = job
    r_parts, r_mats, _ = realize(prefix + target)
    R = list(zip(r_parts[n_prefix:], r_mats[n_prefix:]))
    out = []
    for t in texts:
        g_parts, g_mats, _ = realize(prefix + t)
        if len(g_parts) <= n_prefix:
            out.append((False, False, False, False))
            continue
        gp, gm = g_parts[n_prefix], g_mats[n_prefix]
        pose = [rp.part_id == gp.part_id and same_pose(gp.part_id, gm, rm) for rp, rm in R]
        colour = [ok and rp == gp for ok, (rp, _) in zip(pose, R)]
        out.append((any(colour), any(pose), bool(colour and colour[0]), bool(pose and pose[0])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    run = Path(args.run)
    dev = {r["pair"]: r for r in read_jsonl(args.dev)}
    files = sorted((run / "dev").glob("step-*.jsonl"), key=lambda p: int(p.stem.split("-")[1]))
    names = ("first part right (colour)", "first part right (pose only)", "next part right (colour)",
             "next part right (pose only)")
    result = {}
    with get_context("spawn").Pool(args.workers) as pool:
        for f in files:
            rows = read_jsonl(f)
            jobs = [(dev[r["pair"]]["prefix"], dev[r["pair"]]["target"], r["n_prefix"],
                     [s["text"] for s in r["samples"]]) for r in rows]
            per = pool.map(first_part, jobs, chunksize=4)
            flat = [x for p in per for x in p]
            jump = [x for r, p in zip(rows, per) if r["jump"] for x in p]
            step = int(f.stem.split("-")[1])
            result[step] = {n: sum(x[i] for x in flat) / len(flat) for i, n in enumerate(names)}
            result[step]["first part right (pose only), jump"] = sum(x[1] for x in jump) / max(1, len(jump))
            print(step, json.dumps({k: round(v, 4) for k, v in result[step].items()}), flush=True)
    (run / "first_part.json").write_text(json.dumps(result, indent=1) + "\n")
    cols = list(next(iter(result.values())))
    print("\n| step | " + " | ".join(cols) + " |\n|---:|" + "---:|" * len(cols))
    for s, v in result.items():
        print(f"| {s} | " + " | ".join(f"{v[c]:.3f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
