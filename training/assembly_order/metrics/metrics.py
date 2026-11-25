"""The per-completion quantities behind the report's metrics.

A completion is scored against the held-out structure M in the frame of the shared prefix
(brickicl.eval.match: both texts realize with the prefix parts in the same place).

  n_generated   n_p: parts the model wrote after the prefix that the parser could read
  m_pose        m: one-to-one matches with M on part type, position and orientation; colour is ignored
  m_pose_color  the same with the colour required as well (the count steps 5 and 8 call "placed")
  parse_error   None when the parser reads prefix + completion from the first part to the last

Pose F1 of one completion is 2 m / (n_p + n_t), with n_t = |M|.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from brickicl.eval.match import realize, same_pose


def _matched(ok):
    return int(ok[linear_sum_assignment(-ok.astype(int))].sum()) if ok.size else 0


def score_completion(prefix, target, completion, n_prefix, ref=None):
    """ref: optional cached realize(prefix + target)."""
    r_parts, r_mats, r_err = ref or realize(prefix + target)
    assert r_err is None, r_err
    g_parts, g_mats, g_err = realize(prefix + completion)
    R = list(zip(r_parts[n_prefix:], r_mats[n_prefix:]))
    G = list(zip(g_parts[n_prefix:], g_mats[n_prefix:]))
    pose = np.zeros((len(G), len(R)), dtype=bool)
    both = np.zeros_like(pose)
    for i, (gp, gm) in enumerate(G):
        for j, (rp, rm) in enumerate(R):
            if gp.part_id == rp.part_id and same_pose(gp.part_id, gm, rm):
                pose[i, j] = True
                both[i, j] = gp.color == rp.color
    return {"parse_error": g_err, "n_generated": len(G), "m_pose": _matched(pose), "m_pose_color": _matched(both)}


def pose_f1(sample, n_target, key="m_pose"):
    return 2 * sample[key] / (sample["n_generated"] + n_target)


def parses(sample):
    """The whole work parses: something was generated and the parser reads it to the last part."""
    return sample["parse_error"] is None and sample["n_generated"] >= 1
