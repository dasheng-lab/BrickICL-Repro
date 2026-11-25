"""Build single-path ICL completion data from BrickNet graph splits.

No complete demonstration is prepended. A row is prompt=visible assembly path,
completion=the held-out pendant copy. This writes an intermediate candidate
pool; finalize_splits.py assigns final post-training splits and moves official
val candidates to the previously explored regression pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from bricknet import parse_sample, score_text
from transformers import AutoTokenizer

from brickicl.sampling.build import read_graphs
from brickicl.sampling.edits import stack_joint, stretch
from brickicl.sampling.masked_walk import masked_walk
from brickicl.sampling.missing import pendant_copies


def fingerprint(graph):
    """Conservative graph invariant for grouping exact and many near duplicates."""
    # Ignore color so recolors of the same assembly cannot cross a split.
    nodes = Counter(map(int, graph.part_ids))
    edges = Counter()
    for edge in graph.edges:
        a, b = int(edge["a"]), int(edge["b"])
        key = (
            int(graph.part_ids[a]), int(graph.part_ids[b]),
            str(edge["family"]), int(edge["a_conn"]), int(edge["b_conn"]),
            bool(edge["flip"]), round(float(edge["yaw"]), 3),
            tuple(round(float(x), 3) for x in edge["rot"]),
        )
        edges[key] += 1
    payload = repr((sorted(nodes.items()), sorted(edges.items()))).encode()
    return hashlib.sha256(payload).hexdigest()


def known_development_sources(path):
    if not path or not Path(path).exists():
        return set()
    return {int(json.loads(line)["source"]) for line in open(path, encoding="utf-8") if line.strip()}


def val_partition(graphs, known):
    fingerprints = [fingerprint(g) for g in graphs]
    groups = defaultdict(list)
    for source, fp in enumerate(fingerprints):
        groups[fp].append(source)
    split_of = {}
    for fp, sources in groups.items():
        # A source already used in earlier long-leg experiments cannot enter test.
        dev = bool(known.intersection(sources)) or int(fp[:8], 16) % 5 == 0
        for source in sources:
            split_of[source] = "dev" if dev else "test"
    return fingerprints, split_of


def make_row(graph, source, split, group, target_index, seed_index, condition, r,
             tokenizer, max_tokens, diagnostics):
    target = list(group.copies[target_index])
    others = [j for j in range(len(group.copies)) if j != target_index]
    other_nodes = sorted(v for j in others for v in group.copies[j])
    seed = source * 1009 + target_index * 41 + seed_index * 7
    if condition == "deleted":
        edited, heldout, excluded = graph, target, other_nodes
        blocks = []
    elif condition == "stretched":
        joints = [stack_joint(graph, c, a) for c, a in zip(group.copies, group.anchors)]
        if not all(joints) or len(graph.part_ids) + r * len(joints) > 100:
            diagnostics["no_stretch_joint_or_too_large"] += 1
            return None
        edited, added = stretch(graph, joints, r)
        heldout = target + added[target_index]
        excluded = []
        blocks = [list(group.copies[j]) + added[j] for j in others]
    else:
        edited, heldout, excluded = graph, target, []
        blocks = [group.copies[j] for j in others]
    if len(edited.part_ids) - len(excluded) > 100:
        diagnostics["too_many_parts"] += 1
        return None
    try:
        sample = masked_walk(edited, heldout, exclude=excluded, seed=seed, blocks=blocks)
        full = sample.prefix + sample.target
        parsed = parse_sample(full)
        if parsed.error is not None or len(parsed.tree.parts) != len(sample.walk.order):
            diagnostics["parse_error"] += 1
            return None
        if condition == "stretched" and score_text(full)[2]:
            diagnostics["stretched_collision"] += 1
            return None
        n_tokens = len(tokenizer.encode(full, add_special_tokens=False)) + 1
        if n_tokens > max_tokens:
            diagnostics["token_limit"] += 1
            return None
        return {
            "split": split, "source_split": "sft" if split == "train" else "val",
            "source": source, "condition": condition, "strategy": "repeated_pendant",
            "pair": f"{split}-{source}-{target_index}-{seed_index}",
            "target_copy": target_index, "walk_seed": seed_index, "r": r if condition == "stretched" else 0,
            "n_copies": len(group.copies), "n_prefix": sample.walk.n_prefix,
            "k": len(heldout), "tokens_with_eos": n_tokens,
            "prompt": sample.prefix, "completion": sample.target,
        }
    except (ValueError, AssertionError, KeyError, IndexError) as exc:
        diagnostics[type(exc).__name__] += 1
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-graphs", required=True)
    ap.add_argument("--val-graphs", required=True)
    ap.add_argument("--known-val-sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-train-rows", type=int, default=100000)
    ap.add_argument("--max-tokens", type=int, default=2816)
    ap.add_argument("--walk-seeds", type=int, default=2)
    ap.add_argument("--target-copies", type=int, default=2)
    ap.add_argument("--limit-sft-graphs", type=int, default=0, help="smoke test only")
    ap.add_argument("--resume", action="store_true", help="resume an interrupted build; regenerate its last train source")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists():
        raise RuntimeError("build already completed; choose another output directory")
    if not args.resume and (out / "train.jsonl").exists() and (out / "train.jsonl").stat().st_size:
        raise RuntimeError("candidate data already exists; use --resume for an interrupted build")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    val_graphs = read_graphs(args.val_graphs)
    known = known_development_sources(args.known_val_sources)
    val_fp, val_splits = val_partition(val_graphs, known)
    forbidden = set(val_fp)
    sft_graphs = read_graphs(args.sft_graphs)
    if args.limit_sft_graphs:
        sft_graphs = sft_graphs[:args.limit_sft_graphs]
    counters, skipped = Counter(), Counter()
    sources = defaultdict(set)
    seen_rows = set()
    if args.resume:
        path = out / "train.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        # The previous process may have stopped midway through its last source.
        last_source = rows[-1]["source"] if rows else None
        with path.with_suffix(".resume.tmp").open("w", encoding="utf-8") as stream:
            for row in rows:
                if row["source"] == last_source:
                    continue
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen_rows.add(hashlib.sha256((row["prompt"] + row["completion"]).encode()).digest())
                sources["train"].add(row["source"])
                counters["train"] += 1
                counters[f"train:{row['condition']}"] += 1
        path.with_suffix(".resume.tmp").replace(path)
        del rows
        print(f"Resuming with {counters['train']} rows from {len(sources['train'])} sources", flush=True)
    handles = {name: (out / f"{name}.jsonl").open("a" if args.resume and name == "train" else "w", encoding="utf-8")
               for name in ("train", "dev", "test")}
    try:
        for split, graphs in (("train", sft_graphs), ("val", val_graphs)):
            ids = range(len(graphs))
            if split == "train":
                # Stable pseudorandom order keeps a row cap from favoring low source IDs.
                ids = sorted(ids, key=lambda i: hashlib.sha256(f"sft:{i}".encode()).digest())
            for source in ids:
                if split == "train" and source in sources["train"]:
                    continue
                if split == "train" and counters["train"] >= args.max_train_rows:
                    break
                graph = graphs[source]
                if len(graph.part_ids) > 100 or len(graph.components) != 1:
                    skipped["size_or_components"] += 1
                    continue
                if split == "train" and fingerprint(graph) in forbidden:
                    skipped["val_fingerprint_overlap"] += 1
                    continue
                groups = [g for g in pendant_copies(graph, max_size=8, min_copies=2)
                          if 2 <= g.size <= 8]
                if not groups:
                    skipped["no_repeated_pendant"] += 1
                    continue
                group = groups[0]
                try:
                    original = masked_walk(graph, group.copies[0], seed=0)
                    if score_text(original.prefix + original.target)[2]:
                        skipped["source_collision"] += 1
                        continue
                except (ValueError, AssertionError, KeyError, IndexError):
                    skipped["source_invalid"] += 1
                    continue
                output_split = split if split == "train" else val_splits[source]
                for target_index in range(min(args.target_copies, len(group.copies))):
                    for seed_index in range(args.walk_seeds):
                        r = 1 + (source + target_index + seed_index) % 2
                        for condition in ("original", "deleted", "stretched"):
                            row = make_row(graph, source, output_split, group, target_index,
                                           seed_index, condition, r, tok, args.max_tokens, skipped)
                            if row is None:
                                continue
                            digest = hashlib.sha256((row["prompt"] + row["completion"]).encode()).digest()
                            if digest in seen_rows:
                                skipped["duplicate_text"] += 1
                                continue
                            seen_rows.add(digest)
                            handles[output_split].write(json.dumps(row, ensure_ascii=False) + "\n")
                            counters[output_split] += 1
                            counters[f"{output_split}:{condition}"] += 1
                            sources[output_split].add(source)
    finally:
        for handle in handles.values():
            handle.close()
    metadata = {
        "schema": 1, "task": "single_path_icl_completion", "model_sizes": ["0.6B", "1.7B", "4B"],
        "upstream_splits": {"train": "sft", "dev": "val", "test": "val"},
        "known_val_sources_forced_to_dev": sorted(known),
        "max_tokens_with_eos": args.max_tokens, "max_train_rows": args.max_train_rows,
        "walk_seeds": args.walk_seeds, "target_copies": args.target_copies,
        "counts": dict(counters), "source_counts": {k: len(v) for k, v in sources.items()},
        "skipped": dict(skipped),
        "resumed": args.resume, "skipped_counts_scope": "resume pass only" if args.resume else "whole build",
        "limitations": "SFT graphs were seen by BrickNet pretraining; val test is held out from upstream training. Fingerprint screening catches exact and many coarse structural duplicates, not all near duplicates.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
