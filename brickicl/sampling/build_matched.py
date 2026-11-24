"""Build G -> G/M examples matching each old single-model condition exactly."""
import argparse
import json
import lzma
from pathlib import Path

from bricknet import parse_sample
from transformers import AutoTokenizer

from brickicl.eval.length_probe import shared_head


CONDITIONS = {"original", "stretched", "ablated"}


def pair_indices(pair):
    """Return the target-copy and walk-seed indexes encoded in source-target-seed."""
    _, target, walk_seed = pair.rsplit("-", 2)
    return int(target), int(walk_seed)


def parse_walk_seeds(value):
    seeds = tuple(dict.fromkeys(int(seed.strip()) for seed in value.split(",") if seed.strip()))
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one comma-separated walk seed")
    return seeds


def read_captions(path):
    opener = lzma.open if str(path).endswith(".xz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return {int(row["id"]): row["caption"].strip(" ")
                for row in map(json.loads, stream) if row["caption"].strip(" ")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-demo-output", help="write identical queries without the complete-G demonstration")
    ap.add_argument("--sft-output", help="write caption + incomplete-query prompts for BrickNet-SFT")
    ap.add_argument("--captions", help="caption jsonl(.xz), required with --sft-output")
    ap.add_argument("--max-tokens", type=int, default=6400)
    ap.add_argument("--target-index", type=int, default=0,
                    help="held-out copy index to keep fixed while varying the walk")
    ap.add_argument("--walk-seeds", type=parse_walk_seeds, default=(0,),
                    help="comma-separated walk-seed indexes from the pair suffix (default: 0)")
    args = ap.parse_args()
    if args.sft_output and not args.captions:
        ap.error("--captions is required with --sft-output")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    rows = [json.loads(s) for s in open(args.input)]
    eligible_sources = {row["source"] for row in rows
                        if pair_indices(row["pair"])[0] == args.target_index}
    selected = []
    for row in rows:
        target, walk_seed = pair_indices(row["pair"])
        if target == args.target_index and walk_seed in args.walk_seeds:
            selected.append(row | {"walk_seed_index": walk_seed})
    units = {}
    for row in selected:
        key = row["source"], row["pair"]
        if row["condition"] in units.setdefault(key, {}):
            raise ValueError(f"duplicate condition for {row['pair']}: {row['condition']}")
        units[key][row["condition"]] = row
    bad = {pair: sorted(CONDITIONS - set(group)) for (_, pair), group in units.items()
           if set(group) != CONDITIONS}
    if bad:
        raise ValueError(f"incomplete condition triples: {bad}")
    by_source = {}
    for (source, _), group in units.items():
        by_source.setdefault(source, set()).add(next(iter(group.values()))["walk_seed_index"])
    expected_seeds = set(args.walk_seeds)
    missing = {source: sorted(expected_seeds - by_source.get(source, set()))
               for source in eligible_sources if by_source.get(source, set()) != expected_seeds}
    if missing:
        raise ValueError(f"sources missing requested walk seeds: {missing}")
    condition_order = ("original", "stretched", "ablated")
    rows = [group[condition] for key, group in sorted(units.items()) for condition in condition_order]
    if not rows:
        raise ValueError("no rows matched --target-index and --walk-seeds")
    output = []
    for r in rows:
        head = shared_head(r, 1)
        query = r["prefix"] + head
        expected = "long" if r["condition"] == "stretched" else "short"
        demo = r["prefix"] + r[f"target_{expected}"]
        assert demo.startswith(query)
        assert parse_sample(demo).error is None
        assert parse_sample(query).error is None
        short, long = (r[f"target_{k}"][len(head):] for k in ("short", "long"))
        prompt = demo + tok.eos_token + query
        max_len = max(len(tok.encode(prompt + target, add_special_tokens=False))
                      for target in (short, long))
        if max_len > args.max_tokens:
            raise ValueError(f"{r['pair']} {r['condition']}: {max_len} tokens")
        output.append({
            "source": r["source"], "pair": r["pair"], "condition": r["condition"],
            "expected": expected, "r": r["r"], "seed": r.get("seed"),
            "walk_seed_index": r["walk_seed_index"],
            "n_prefix": r["n_prefix"] + 1,
            "k_short": r["k_short"] - 1, "k_long": r["k_long"] - 1,
            "demo_text": demo, "query_prefix": query,
            "target_short": short, "target_long": long,
            "prompt": prompt, "max_seq_tokens": max_len,
        })
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        for r in output: f.write(json.dumps(r) + "\n")
    if args.no_demo_output:
        control = Path(args.no_demo_output)
        control.parent.mkdir(parents=True, exist_ok=True)
        with control.open("w") as f:
            for r in output:
                f.write(json.dumps(r | {"demo_text": "", "prompt": r["query_prefix"]}) + "\n")
    if args.sft_output:
        captions = read_captions(args.captions)
        missing = sorted({r["source"] for r in output} - set(captions))
        if missing:
            raise ValueError(f"captions missing sources: {missing}")
        sft_rows = []
        for r in output:
            caption = captions[r["source"]]
            prompt = caption + "\n" + r["query_prefix"]
            max_len = max(len(tok.encode(prompt + r[f"target_{kind}"], add_special_tokens=False))
                          for kind in ("short", "long"))
            if max_len > args.max_tokens:
                raise ValueError(f"SFT {r['pair']} {r['condition']}: {max_len} tokens")
            sft_rows.append(r | {
                "caption": caption,
                "demo_text": "",
                "prompt": prompt,
                "max_seq_tokens": max_len,
                "context_setting": "sft_caption_plus_partial",
            })
        sft = Path(args.sft_output)
        sft.parent.mkdir(parents=True, exist_ok=True)
        with sft.open("w") as f:
            for r in sft_rows:
                f.write(json.dumps(r) + "\n")
    print(len(output), "matched rows from", len({r["source"] for r in output}),
          "sources and", len({r["pair"] for r in output}), "source/walk units; max tokens",
          max(r["max_seq_tokens"] for r in output))


if __name__ == "__main__":
    main()
