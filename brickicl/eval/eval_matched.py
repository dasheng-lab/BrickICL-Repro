"""Evaluate full-G demonstrations matched to the old G/M query contexts."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from brickicl.eval.frozen import load, sample, target_logprobs
from brickicl.eval.length_probe import leg_stack, stacked_part
from brickicl.eval.match import realize, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", action="append", required=True,
                    help="LoRA adapter(s), merged in order (repeat for PT + SFT)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=180000)
    args = ap.parse_args()
    rows = [json.loads(s) for s in open(args.data)]
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair"]].append(row)
    conditions = {"original", "stretched", "ablated"}
    bad = {pair: sorted(r["condition"] for r in group) for pair, group in groups.items()
           if len(group) != 3 or {r["condition"] for r in group} != conditions
           or len({r["source"] for r in group}) != 1}
    if bad:
        raise ValueError(f"invalid source/walk condition triples: {bad}")
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    done_rows = [json.loads(s) for s in dest.open()] if dest.exists() else []
    done_groups = defaultdict(list)
    for row in done_rows:
        done_groups[row["pair"]].append(row)
    completed = {pair for pair, group in groups.items()
                 if len(done_groups[pair]) == len(group)
                 and {r["condition"] for r in done_groups[pair]} == conditions
                 and len({r["source"] for r in done_groups[pair]}) == 1}
    clean_rows = [r for r in done_rows if r["pair"] in completed]
    if len(clean_rows) != len(done_rows):
        done_rows = clean_rows
        with dest.open("w") as f:
            for r in done_rows: f.write(json.dumps(r) + "\n")
    pending = [(pair, group) for pair, group in sorted(groups.items()) if pair not in completed]
    n_sources = len({r["source"] for r in rows})
    print(f"{len(completed)}/{len(groups)} source/walk units done across {n_sources} sources; "
          f"{len(pending)} pending", flush=True)
    if not pending: return
    model, tok = load(args.model, args.lora, "bfloat16", "cuda")
    with dest.open("a") as f:
        for i, (pair, rs) in enumerate(pending, 1):
            rs = sorted(rs, key=lambda r: ("original", "stretched", "ablated").index(r["condition"]))
            source = rs[0]["source"]
            lp = {}
            for kind in ("short", "long"):
                inputs = [{"prefix": r["prompt"], "target": r[f"target_{kind}"]} for r in rs]
                lp[kind], bad = target_logprobs(model, tok, inputs, 3, "cuda")
                if bad: raise RuntimeError(f"source {source}: {bad} token-boundary mismatches")
            prompts = [{"prefix": r["prompt"], "cap": r["k_long"]} for r in rs]
            sample_seed = rs[0].get("seed")
            if sample_seed is None:
                sample_seed = source
            texts = sample(model, tok, prompts, args.n, args.token_budget, sample_seed, "cuda",
                           k_key="cap", extra_parts=4)
            for j, (r, comps) in enumerate(zip(rs, texts)):
                refs = {kind: realize(r["query_prefix"] + r[f"target_{kind}"])
                        for kind in ("short", "long")}
                stack = stacked_part(r)
                leg_start = r["n_prefix"] - 1
                leg_short = leg_stack(r["query_prefix"] + r["target_short"], leg_start, stack)
                leg_long = leg_stack(r["query_prefix"] + r["target_long"], leg_start, stack)
                head = "".join(r["query_prefix"].splitlines(True)[-2:])
                samples = []
                for comp in comps:
                    ss = score(r["query_prefix"], r["target_short"], comp, r["n_prefix"], refs["short"])
                    sl = score(r["query_prefix"], r["target_long"], comp, r["n_prefix"], refs["long"])
                    leg = leg_stack(r["query_prefix"] + comp, leg_start, stack)
                    n_stack = sum(line.split(" ", 1)[-1] == stack for line in (head + comp).splitlines()[0::2])
                    samples.append({
                        "text": head + comp,
                        "raw_completion": comp,
                        "category": "long" if sl.exact else "short" if ss.exact else "other",
                        "n_parts": ss.n_generated + 1,
                        "n_stack": n_stack,
                        "leg": leg,
                        "ended": comp.endswith("\n") and ss.parse_error is None,
                        "parse_error": ss.parse_error,
                    })
                n = len(samples) or 1
                cats = Counter(s["category"] for s in samples)
                lp_s = sum(lp["short"][j][x] for x in ("lp_target", "lp_eos"))
                lp_l = sum(lp["long"][j][x] for x in ("lp_target", "lp_eos"))
                out = {
                    "source": source, "pair": r["pair"], "condition": r["condition"],
                    "expected": r["expected"], "model": args.model, "lora": args.lora,
                    "seed": r.get("seed"), "walk_seed_index": r.get("walk_seed_index"),
                    "r": r["r"], "prime": 1, "n_prefix": r["n_prefix"] - 1,
                    "k_short": r["k_short"] + 1, "k_long": r["k_long"] + 1,
                    "lp_short": lp_s, "lp_long": lp_l, "log_odds": lp_l - lp_s,
                    "frac_long": cats["long"] / n, "frac_short": cats["short"] / n,
                    "mean_stack": sum(s["n_stack"] for s in samples) / n,
                    "mean_parts": sum(s["n_parts"] for s in samples) / n,
                    "leg_short": leg_short, "leg_long": leg_long,
                    "leg_extra": sum(s["leg"] - leg_short for s in samples) / n,
                    "leg_is_short": sum(s["leg"] == leg_short for s in samples) / n,
                    "leg_is_long": sum(s["leg"] == leg_long for s in samples) / n,
                    "leg_ge_long": sum(s["leg"] >= leg_long for s in samples) / n,
                    "samples": samples,
                }
                f.write(json.dumps(out) + "\n")
            f.flush()
            print(f"{args.model}: {i}/{len(pending)} source/walk units, pair {pair}", flush=True)


if __name__ == "__main__":
    main()
