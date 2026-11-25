"""Pose-based completion reward used by the GRPO trainer.

Matching uses the shared assembly-order metrics implementation.
"""

import importlib.util
import os
import sys
from pathlib import Path

SRC = Path(os.environ.get("SRC", str(Path(__file__).resolve().parents[4])))
METRICS_DIR = SRC / "scripts/train/assembly_order/metrics"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brickicl.eval.match import realize  # noqa: E402,F401  (re-exported: callers cache realize(prefix + target))

_spec = importlib.util.spec_from_file_location("assembly_order_metrics", METRICS_DIR / "metrics.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
score_completion, pose_f1, parses = _m.score_completion, _m.pose_f1, _m.parses

# Source files whose content determines the reward.
EXTERNAL = [
    METRICS_DIR / "metrics.py",
    SRC / "src" / "brickicl" / "eval" / "match.py",
    SRC / "third_party" / "bricknet" / "BrickNet" / "src" / "bricknet" / "score.py",
    SRC / "third_party" / "bricknet" / "BrickNet" / "src" / "bricknet" / "collision.py",
    SRC / "third_party" / "bricknet" / "BrickNet" / "src" / "bricknet" / "tree.py",
]


def f_beta(sample, n_target, beta=1.0, key="m_pose"):
    """(1 + b^2) m / (b^2 n_t + n_p); beta=1 gives pose F1."""
    b2 = beta * beta
    return (1 + b2) * sample[key] / (b2 * n_target + sample["n_generated"])


def completion(prefix, target, text, n_prefix, k, ref=None, beta=1.0, fail=0.0):
    """Quantities of one completion and its reward. ref: optional cached realize(prefix + target).

    reward = pose F-beta if the text parses, else `fail`."""
    s = score_completion(prefix, target, text, n_prefix, ref)
    s["f1"] = pose_f1(s, k)
    s["parses"] = parses(s)
    s["exact"] = s["parses"] and s["n_generated"] == k and s["m_pose"] == k
    s["reward"] = f_beta(s, k, beta) if s["parses"] else fail
    return s
