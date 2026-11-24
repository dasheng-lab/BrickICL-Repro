"""Check the pools written by build_data.py. Exits non-zero when a check fails.

  pairing    masked row i is plain row i with the block of M moved to the end
  disjoint   no model appears in two pools (train / dev)
  unique     no path text repeats within an arm
  sequences  a random sample parses back; plain is a depth-first preorder, masked leaves it at
             most once, at the first part of M
  tokens     length under the Qwen3 tokenizer (skipped with --no-tokens)

Usage (from the repo root, in the bricknet env):
  python experiments/assembly_order/data_construction/verify_data.py runs/assembly_order/sft_train_v1 runs/assembly_order/sft_dev_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from bricknet import parse_sample


def off_path(tree):
    """Positions whose parent is neither the previous part nor one of its ancestors (none in a preorder)."""
    par = [None] + [e.parent for e in tree.edges]
    out = []
    for i in range(2, len(par)):
        u = i - 1
        while u is not None and u != par[i]:
            u = par[u]
        if u is None:
            out.append(i)
    return out


def load(d, arm):
    with open(Path(d) / f"{arm}.jsonl") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+", help="output directories of build_data.py")
    ap.add_argument("--sample", type=int, default=2000, help="sequences to parse back per pool")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="tokenizer")
    ap.add_argument("--max-len", type=int, default=6400)
    ap.add_argument("--no-tokens", action="store_true")
    args = ap.parse_args()

    tok = None
    if not args.no_tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)

    failures, sources = [], {}

    def check(ok, msg):
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    for d in args.pools:
        masked, plain = load(d, "masked"), load(d, "plain")
        print(f"{d}: {len(masked)} masked, {len(plain)} plain")
        check(len(masked) == len(plain) > 0, "both arms have the same number of rows")
        bad = 0
        for m, p in zip(masked, plain):
            i, k, o = p["block_start"], p["k"], p["order"]
            bad += (
                m["source"] != p["source"]
                or m["sample_index"] != p["sample_index"]
                or m["M"] != o[i : i + k]
                or m["order"] != o[:i] + o[i + k :] + m["M"]
                or i + k >= len(o)
            )
        check(bad == 0, f"pairing: masked is plain with the block of M moved to the end ({bad} mismatches)")
        for arm, rows in (("masked", masked), ("plain", plain)):
            n_dup = len(rows) - len({r["path"] for r in rows})
            check(n_dup == 0, f"unique: no repeated path in {arm} ({n_dup} repeats)")
        cross = len({r["path"] for r in masked} & {r["path"] for r in plain})
        print(f"  note  {cross} masked paths also occur as a plain path of another pair")

        rng = np.random.default_rng(0)
        n_bad = 0
        for i in rng.choice(len(masked), size=min(args.sample, len(masked)), replace=False).tolist():
            m, p = masked[i], plain[i]
            rm, rp = parse_sample(m["path"]), parse_sample(p["path"])
            ok = rm.error is None and rp.error is None
            ok = ok and len(rm.tree.parts) == len(rp.tree.parts) == m["nodes"] == p["nodes"] <= 100
            ok = ok and m["order"][m["n_prefix"] :] == m["M"] and len(m["M"]) == m["k"]
            ok = ok and m["path"].count("\n") == 2 * m["nodes"] - 1 and m["path"].endswith("\n")
            ok = ok and off_path(rp.tree) == [] and off_path(rm.tree) == ([m["n_prefix"]] if m["jump"] else [])
            n_bad += not ok
        check(n_bad == 0, f"sequences: sampled pairs parse back, preorder except at M ({n_bad} bad)")

        srcs = {(r["split"], r["source"]) for r in masked}  # `source` is an index within its split
        for other, s in sources.items():
            check(not (srcs & s), f"disjoint: no model shared with {other} ({len(srcs & s)} shared)")
        sources[d] = srcs

        k = np.array([r["k"] for r in masked])
        n = np.array([r["nodes"] for r in masked])
        per = np.unique([r["source"] for r in masked], return_counts=True)[1]
        print(f"  models {len(srcs)}; pairs per model mean {per.mean():.1f}, median {np.median(per):.0f}, max {per.max()}")
        print(f"  masked rows with an off-path attachment: {np.mean([r['jump'] for r in masked]):.1%}")
        print(f"  parts per sequence mean {n.mean():.1f}, median {np.median(n):.0f}; "
              f"|M| mean {k.mean():.1f}, median {np.median(k):.0f}; M share of the sequence {np.mean(k / n):.2f}")
        if tok is not None:
            for arm, rows in (("masked", masked), ("plain", plain)):
                t = np.array([len(x) for x in tok([r["path"] for r in rows], add_special_tokens=False)["input_ids"]])
                print(f"  {arm:6s} tokens: total {t.sum():,}, mean {t.mean():.0f}, median {np.median(t):.0f}, "
                      f"p90 {np.percentile(t, 90):.0f}, max {t.max()}")
                check(int((t > args.max_len).sum()) == 0, f"tokens: no {arm} sequence over {args.max_len}")

    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECKS FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
