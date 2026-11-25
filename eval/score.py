"""Score sampled LEGO continuations with Pose F1, Exact@N, and paired DLO."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.optimize import linear_sum_assignment
from brickicl.eval.match import realize, same_pose


def parts_and_poses(text):
    return realize(text)


def pose_f1(predicted, predicted_poses, target, target_poses):
    matrix = np.zeros((len(predicted), len(target)), dtype=np.int8)
    for i, a in enumerate(predicted):
        for j, b in enumerate(target):
            if a == b and same_pose(a.part_id, predicted_poses[i], target_poses[j]):
                matrix[i, j] = 1
    hits = int(matrix[linear_sum_assignment(-matrix)].sum()) if matrix.size else 0
    denominator = len(predicted) + len(target)
    return 2 * hits / denominator if denominator else 1.0


def log_odds(logp):
    if logp >= 0:
        raise ValueError("Expected a negative log probability")
    return logp - math.log1p(-math.exp(logp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    scored = []
    paired = defaultdict(dict)
    for row in rows:
        prefix = row["prefix"]
        target_text = row.get("target")
        if target_text is None:
            raise ValueError(f"Target missing for {row['id']}")
        context, _, context_error = parts_and_poses(prefix)
        target, target_poses, target_error = parts_and_poses(prefix + target_text)
        if context_error or target_error:
            raise ValueError(f"Invalid reference or target for {row['id']}")
        cut = len(context)
        gold, gold_poses = target[cut:], target_poses[cut:]
        samples = []
        for text in row["samples"]:
            predicted, poses, error = parts_and_poses(prefix + text)
            candidate = predicted[cut:cut + len(gold)]
            candidate_poses = poses[cut:cut + len(gold)]
            f1 = pose_f1(candidate, candidate_poses, gold, gold_poses)
            samples.append({"pose_f1": f1, "exact": error is None and len(predicted) - cut == len(gold) and f1 == 1.0})
        scored.append({"id": row["id"], "source_key": row["source_key"],
                       "condition": row.get("condition"), "samples": samples})
        if row.get("condition") in ("original", "stretched") and "target_logp" in row:
            paired[row["source_key"]][row["condition"]] = log_odds(row["target_logp"])
    summary = {
        "rows": len(scored),
        "pose_f1": mean(s["pose_f1"] for row in scored for s in row["samples"]),
        "exact_at_n": mean(any(s["exact"] for s in row["samples"]) for row in scored),
    }
    complete = [v["stretched"] - v["original"] for v in paired.values() if set(v) == {"original", "stretched"}]
    if complete:
        summary["delta_log_odds"] = mean(complete)
        summary["paired_sources"] = len(complete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": scored}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
