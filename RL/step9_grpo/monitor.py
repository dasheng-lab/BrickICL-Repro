"""Watch a GRPO run and return when something needs a look. (CPU, any env)

Polls <run>/log.jsonl and <run>/dev.jsonl every --every seconds and exits, printing a short report, when
  - a new dev evaluation is written (more than --dev-seen lines in dev.jsonl),
  - the trainer is no longer running (finished or crashed: the tail of --log is printed),
  - the mean of the last --window steps crosses an alarm: entropy < --min-entropy, KL > --max-kl,
    rollout n_p / n_t < --min-np-nt, groups kept < --min-groups, a non-finite loss, or (logs with these fields,
    GRPO v2) empty completions > --max-empty or stopped-before-target > --max-short.
Exit code 0 for a dev line or the end of the run, 2 for an alarm.

Usage (from ~/RL):
  python step9_grpo/monitor.py --run runs/r1/train/grpo --log logs/r1_grpo.log --dev-seen 1
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def lines(p):
    try:
        return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def report(log, dev, window):
    out = []
    if log:
        last = log[-window:]
        mean = lambda k: sum(x[k] for x in last) / len(last)  # noqa: E731
        out.append(f"step {log[-1]['step']}: last {len(last)} steps mean reward {mean('rollout_reward'):.4f} "
                   f"kept {mean('groups_kept'):.1f} n_p/n_t {mean('rollout_np_over_nt'):.2f} "
                   f"parses {mean('rollout_parses'):.3f} eos {mean('rollout_eos'):.3f} kl {mean('kl'):.5f} "
                   f"ent {mean('entropy'):.3f} step {mean('step_s'):.0f}s")
        for a, b in ((1, 25), (26, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)):
            seg = [x for x in log if a <= x["step"] <= b]
            if seg:
                out.append(f"  steps {a}-{b}: reward {sum(x['rollout_reward'] for x in seg) / len(seg):.4f} "
                           f"n_p/n_t {sum(x['rollout_np_over_nt'] for x in seg) / len(seg):.2f}")
    for d in dev:
        out.append(f"  dev step {d['step']}: reward {d['reward']:.4f} f1 {d['f1']:.4f} exact {d['exact']:.4f} "
                   f"parses {d['parses']:.3f} n_p/n_t {d['np_over_nt']:.2f} jump {d['reward, jump']:.4f}"
                   + (f" empty {d['empty']:.3f} short {d['short']:.3f}" if "empty" in d else "")
                   + (f" | stretch: hit {d['stretch hit, stretched']:.3f} / {d['stretch hit, original']:.3f} "
                      f"dlogodds {d['stretch dlogodds']:+.2f} short {d['stretch short']:.3f}"
                      if "stretch dlogodds" in d else ""))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--dev-seen", type=int, default=0)
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--min-entropy", type=float, default=0.08)
    ap.add_argument("--max-kl", type=float, default=0.05)
    ap.add_argument("--min-np-nt", type=float, default=0.8)
    ap.add_argument("--min-groups", type=float, default=8)
    ap.add_argument("--max-empty", type=float, default=0.25, help="r1: 0.17 at the start, 0.27 when it was stopped")
    ap.add_argument("--max-short", type=float, default=0.5)
    args = ap.parse_args()
    run = Path(args.run)
    while True:
        log, dev = lines(run / "log.jsonl"), lines(run / "dev.jsonl")
        alive = subprocess.run(["pgrep", "-f", f"grpo.py .*--out {run.resolve()}"], capture_output=True).returncode == 0
        alarms = []
        if len(log) >= args.window:
            last = log[-args.window:]
            mean = lambda k: sum(x[k] for x in last) / len(last)  # noqa: E731
            if any(not math.isfinite(x["loss"]) for x in last):
                alarms.append("non-finite loss")
            if mean("entropy") < args.min_entropy:
                alarms.append(f"entropy {mean('entropy'):.3f} < {args.min_entropy}")
            if mean("kl") > args.max_kl:
                alarms.append(f"kl {mean('kl'):.4f} > {args.max_kl}")
            if mean("rollout_np_over_nt") < args.min_np_nt:
                alarms.append(f"n_p/n_t {mean('rollout_np_over_nt'):.2f} < {args.min_np_nt}")
            if mean("groups_kept") < args.min_groups:
                alarms.append(f"groups kept {mean('groups_kept'):.1f} < {args.min_groups}")
            if all("rollout_empty" in x for x in last) and mean("rollout_empty") > args.max_empty:
                alarms.append(f"empty {mean('rollout_empty'):.3f} > {args.max_empty}")
            if all("rollout_short" in x for x in last) and mean("rollout_short") > args.max_short:
                alarms.append(f"short {mean('rollout_short'):.3f} > {args.max_short}")
        event = None
        if alarms:
            event = "ALARM: " + "; ".join(alarms)
        elif len(dev) > args.dev_seen:
            event = f"DEV: new dev evaluation (dev lines {len(dev)})"
        elif not alive:
            text = Path(args.log).read_text(errors="replace").splitlines() if Path(args.log).exists() else []
            tail = [x[:300] for x in text if not x.startswith(("WARNING", "INFO", "pybullet"))][-15:]
            event = "STOPPED: trainer not running\n" + "\n".join(tail)
        if event:
            print(time.strftime("%F %T UTC", time.gmtime()), event)
            print(report(log, dev, args.window))
            sys.exit(2 if alarms else 0)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
