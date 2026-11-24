"""CPU checks of the GRPO v2 pieces, and what the v2 reward says about r1. (CPU, bricknet env)

  1. beta=1, fail=0 reproduces r1's reward exactly (rescoring r1's dev samples of one checkpoint).
  2. the r1 dev checkpoints under r1's and v2's reward, with the empty and stopped-short rates: does the v2 reward
     rank the late, short checkpoints lower?
  3. the stretched dev prompts scored with known answers (own target, target cut short, empty): rewards, long hit
     and sunshq's Δlog-odds come out as they must.
  4. the three loss normalisers give the per-token gradient weights they claim (train_step with a stand-in model).

Usage (from ~/RL, `source env.sh`):
  python step9_grpo/check_v2.py --r1 runs/r1/train/grpo --dev runs/dev_test/test.jsonl --stretch-dev runs/stretch/dev.jsonl
"""

import argparse
import contextlib
import math
import sys
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grpo  # noqa: E402
from common import rewards  # noqa: E402
from common.files import read_jsonl  # noqa: E402


def rescore(job):
    row, texts = job
    ref = rewards.realize(row["prefix"] + row["target"])
    return [rewards.completion(row["prefix"], row["target"], t, row["n_prefix"], row["k"], ref)["reward"]
            for t in texts]


def check_r1_reward(run, dev, pool, step=300, rows=200):
    got = read_jsonl(run / "dev" / f"step-{step}.jsonl")[:rows]
    new = pool.map(rescore, [(dev[r["pair"]], [s["text"] for s in r["samples"]]) for r in got], chunksize=4)
    old = [[s["reward"] for s in r["samples"]] for r in got]
    diff = max(abs(a - b) for o, n in zip(old, new) for a, b in zip(o, n))
    print(f"1. beta=1, fail=0 vs r1's stored reward, step {step}, {sum(map(len, old))} samples: max |diff| {diff:.2e}")
    assert diff < 1e-12


def r1_under_v2(run):
    print("\n2. r1 dev checkpoints (600 prompts x 4) under r1's and v2's reward (F2, fail -0.1)")
    print("| step | r1 reward (F1) | v2 reward | empty | parses, stopped before n_t | n_p / n_t |\n|---:|---:|---:|---:|---:|---:|")
    for f in sorted((run / "dev").glob("step-*.jsonl"), key=lambda p: int(p.stem.split("-")[1])):
        rows = read_jsonl(f)
        s = [(r["k"], x) for r in rows for x in r["samples"]]
        f1 = np.mean([rewards.pose_f1(x, k) if rewards.parses(x) else 0.0 for k, x in s])
        v2 = np.mean([rewards.f_beta(x, k, 2.0) if rewards.parses(x) else -0.1 for k, x in s])
        empty = np.mean([x["n_generated"] == 0 for _, x in s])
        short = np.mean([rewards.parses(x) and x["n_generated"] < k for k, x in s])
        npnt = np.mean([x["n_generated"] / k for k, x in s])
        print(f"| {f.stem.split('-')[1]} | {f1:.4f} | {v2:.4f} | {empty:.3f} | {short:.3f} | {npnt:.2f} |")


def check_stretch(rows, pool):
    """Every sample of a stretched row is its own target (all hit), of an original row its own target (none hit,
    since the short copy has fewer of the extra part): Δlog-odds = 2 ln(4.5 / 0.5) per pair."""
    def first_parts(target, n):  # every part of M is two lines (the prefix holds part 1)
        return "".join(target.splitlines(keepends=True)[: 2 * n])

    args = SimpleNamespace(beta=2.0, fail_reward=-0.1)
    comps = [[(r["target"], True)] * 4 for r in rows]
    groups = list(zip(rows, pool.map(grpo.score_group, [(r, c, args.beta, args.fail_reward)
                                                        for r, c in zip(rows, comps)], chunksize=4)))
    assert all(x["reward"] == 1.0 and x["f1"] == 1.0 for _, g in groups for x in g)
    assert all(x["long_hit"] == (r["condition"] == "stretched") for r, g in groups for x in g)
    st = grpo.stretch_stats(groups)
    want = 2 * math.log(4.5 / 0.5)
    print(f"\n3. stretched dev, {len(rows)} rows, own target x 4: {st}")
    assert abs(st["dlogodds"] - want) < 1e-9 and st["hit, stretched"] == 1 and st["hit, original"] == 0
    # cut short by r parts: fewer of the extra part than the long copy unless only other parts were cut;
    # empty -> the fail reward
    st_rows = [r for r in rows if r["condition"] == "stretched" and r["k"] > r["r"]]
    jobs = [(r, [(first_parts(r["target"], r["k"] - r["r"]), True), ("", True)], 2.0, -0.1) for r in st_rows]
    out = pool.map(grpo.score_group, jobs, chunksize=4)
    cut = [g[0] for g in out]
    assert all(x["parses"] and x["n_generated"] == r["k"] - r["r"] for x, r in zip(cut, st_rows))
    assert all(x["long_hit"] == (x["n_extra"] >= r["n_extra_long"]) and x["n_extra"] <= r["n_extra_long"]
               for x, r in zip(cut, st_rows))
    assert all(g[1]["reward"] == -0.1 and g[1]["n_generated"] == 0 for g in out)
    print(f"   cut short by r: reward F2 {np.mean([x['reward'] for x in cut]):.3f}, "
          f"F1 {np.mean([x['f1'] for x in cut]):.3f}, long hit {np.mean([x['long_hit'] for x in cut]):.3f} "
          f"({len(cut)} rows); empty: reward -0.1")


def check_loss_norm():
    import torch

    class Stand(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.zeros(64))

        def get_base_model(self):
            return self

        def disable_adapter(self):
            return contextlib.nullcontext()

    def fake_logps(base, ids, attn, spans, chunk):  # log p of token t = theta[t] - 1, per completion position
        tok = torch.cat([ids[b, s : s + n] for b, s, n in spans])
        return base.theta[tok] - 1.0, torch.zeros(len(tok))

    grpo.token_logps = fake_logps
    batch = [([1, 2], [3], 0.5), ([1, 2], [4, 5, 6, 7], -0.25), ([1], [8, 9], 0.1)]  # (prompt, completion, adv)
    total, n = 7, 3
    print("\n4. loss normalisers: gradient on theta[token] = -A * weight")
    for norm, w in (("batch", lambda c: 1 / total), ("const", lambda c: 1 / 1000), ("seq", lambda c: 1 / (len(c) * n))):
        m = Stand()
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        args = SimpleNamespace(loss_norm=norm, norm_tokens=1000, micro_tokens=8, logit_chunk=4, kl=0.0, max_grad_norm=1e9)
        grad = {}
        orig = opt.step
        opt.step = lambda: grad.update(g=m.theta.grad.clone()) or orig()
        grpo.train_step(m, opt, batch, args, 0, "cpu")
        want = torch.zeros(64)
        for _, c, a in batch:
            for t in c:
                want[t] += -a * w(c)
        err = (grad["g"] - want).abs().max().item()
        print(f"   {norm}: max |grad - expected| {err:.2e}")
        assert err < 1e-7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", default="runs/r1/train/grpo")
    ap.add_argument("--dev", default="runs/dev_test/test.jsonl")
    ap.add_argument("--stretch-dev", default="runs/stretch/dev.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    dev = {r["pair"]: r for r in read_jsonl(args.dev)}
    with get_context("spawn").Pool(args.workers) as pool:
        check_r1_reward(Path(args.r1), dev, pool)
        r1_under_v2(Path(args.r1))
        check_stretch(read_jsonl(args.stretch_dev), pool)
    check_loss_norm()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
