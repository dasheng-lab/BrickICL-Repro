"""Rewards shared by RFT, DPO and GRPO. Thin wrappers over code that already defines the report's metrics:

  completion    Pose F1 = 2 m / (n_p + n_t), colour ignored, from nzd-supertry/step11_report_metrics/metrics.py
                (which matches parts with brickicl.eval.match: same frame, connector positions, 3 LDU, bipartite).
                A completion that does not parse to its last part, or writes nothing, gets reward 0.
                GRPO v2 options: F-beta instead of F1 (beta > 1 weights recall: stopping short costs more than
                writing too much), and a negative reward for a completion that does not parse or is empty.
                beta=1, fail=0 is the reward above, exactly.
  from scratch  the official parse and collision check (bricknet.score.score_text) on the text cut at 100 parts,
                as in step11's cap_generation.py.

Nothing here re-implements matching or collision; step1_checks/ verifies that these wrappers reproduce step 11.
"""

import importlib.util
import os
import sys
from pathlib import Path

SRC = Path(os.environ.get("SRC", "/home/nzd0420/brickICL"))
STEP11 = SRC / "nzd-supertry" / "step11_report_metrics"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bricknet.score import score_text  # noqa: E402

from brickicl.eval.match import realize  # noqa: E402,F401  (re-exported: callers cache realize(prefix + target))

_spec = importlib.util.spec_from_file_location("step11_metrics", STEP11 / "metrics.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
score_completion, pose_f1, parses = _m.score_completion, _m.pose_f1, _m.parses

MAX_PARTS = 100

# files outside ~/RL whose content decides a reward; hashed into every run_config.json (common/io.py)
EXTERNAL = [
    STEP11 / "metrics.py",
    SRC / "brickicl" / "eval" / "match.py",
    SRC / "baseline" / "bricknet" / "BrickNet" / "src" / "bricknet" / "score.py",
    SRC / "baseline" / "bricknet" / "BrickNet" / "src" / "bricknet" / "collision.py",
    SRC / "baseline" / "bricknet" / "BrickNet" / "src" / "bricknet" / "tree.py",
]


def f_beta(sample, n_target, beta=1.0, key="m_pose"):
    """(1 + b^2) m / (b^2 n_t + n_p); beta=1 is step 11's pose_f1 (2 m / (n_p + n_t))."""
    b2 = beta * beta
    return (1 + b2) * sample[key] / (b2 * n_target + sample["n_generated"])


def completion(prefix, target, text, n_prefix, k, ref=None, colour=False, beta=1.0, fail=0.0):
    """Quantities of one completion and its reward. ref: optional cached realize(prefix + target).

    reward = F-beta (pose, or pose and colour with colour=True) if the text parses, else `fail`.
    f1 and f1_color are always the step 11 F1, whatever beta, so runs with different rewards compare."""
    s = score_completion(prefix, target, text, n_prefix, ref)
    s["f1"] = pose_f1(s, k)
    s["f1_color"] = pose_f1(s, k, "m_pose_color")
    s["parses"] = parses(s)
    s["exact"] = s["parses"] and s["n_generated"] == k and s["m_pose_color"] == k
    s["reward_f1"] = (s["f1_color"] if colour else s["f1"]) if s["parses"] else 0.0  # the r1 reward
    s["reward"] = f_beta(s, k, beta, "m_pose_color" if colour else "m_pose") if s["parses"] else fail
    return s


def part_count(prefix, text, n_prefix, part_id):
    """How many parts named part_id (as realize names it) the completion writes; parsed parts only."""
    parts, _, _ = realize(prefix + text)
    return sum(str(p.part_id) == part_id for p in parts[n_prefix:])


def cap(text, max_parts=MAX_PARTS):
    """The first max_parts parts of a path text (part 1 is one line, every later part two)."""
    return "\n".join(text.rstrip("\n").split("\n")[: 2 * max_parts - 1])


def generation(text, collision=True):
    """From-scratch sample: lengths of its valid and collision-free heads, and two candidate rewards.

    reward_frac  collision-free head / parts written: the proposal's "validity" (1.0 for a clean 1-part sample)
    reward_len   collision-free head / 100: favours long clean builds
    Which one to train on is decided in step 3; both are kept.
    """
    t = cap(text)
    n_actions, invalid, collisions = score_text(t, collision)
    valid = n_actions if invalid is None else invalid
    clean = None if collisions is None else min([valid] + list(collisions))
    head = valid if clean is None else clean
    return {
        "n_actions": n_actions,
        "valid_len": valid,
        "collision_free_len": clean,
        "parses": invalid is None,
        "collider_missing": collision and collisions is None,
        "reward_frac": head / n_actions if n_actions else 0.0,
        "reward_len": head / MAX_PARTS,
    }
