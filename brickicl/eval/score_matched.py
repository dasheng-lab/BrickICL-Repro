"""Compare matched G -> G/M samples with the identical old single-model queries."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from brickicl.eval.match import realize, score
from brickicl.eval.report import cluster_ci


METRICS = ("log_odds", "node_f1", "pose_f1", "pose_pref", "exact", "parse", "leg_is_long")
CONDITIONS = ("original", "stretched", "ablated")


def read(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metrics(data, output, matched):
    rows = {(r["pair"], r["condition"]): r for r in read(data)}
    result = defaultdict(dict)
    output_rows = read(output)
    for r in output_rows:
        key = r["pair"], r["condition"]
        if key not in rows:
            continue
        d = rows[key]
        expected = "long" if r["condition"] == "stretched" else "short"
        if matched:
            prefix = d["query_prefix"]
            generated = [s["raw_completion"] for s in r["samples"]]
            n_prefix = d["n_prefix"]
        else:
            prefix = d["prefix"]
            generated = [s["text"] for s in r["samples"]]
            n_prefix = d["n_prefix"]
        f1 = {}
        for kind in ("short", "long"):
            tail = d[f"target_{kind}"]
            k = d[f"k_{kind}"]
            ref = realize(prefix + tail)
            vals = [score(prefix, tail, text, n_prefix, ref) for text in generated]
            f1[kind] = {
                "node": np.mean([2*v.part_hits/(k+v.n_generated) for v in vals]),
                "pose": np.mean([2*v.placed/(k+v.n_generated) for v in vals]),
                "exact": np.mean([v.exact for v in vals]),
                "parse": np.mean([v.parse_error is None for v in vals]),
            }
        result[r["pair"]][r["condition"]] = {
            "source": r["source"],
            "seed": r.get("seed"),
            "walk_seed_index": r.get("walk_seed_index"),
            "log_odds": r["log_odds"],
            "node_f1": f1[expected]["node"],
            "pose_f1": f1[expected]["pose"],
            "exact": f1[expected]["exact"],
            "parse": f1[expected]["parse"],
            "pose_pref": f1["long"]["pose"] - f1["short"]["pose"],
            "leg_is_long": r["leg_is_long"],
        }
    model = output_rows[0].get("model", Path(output).name) if output_rows else Path(output).name
    return result, model


def show(label, groups):
    groups = {pair: rows for pair, rows in groups.items() if set(CONDITIONS) <= set(rows)}
    sources = [rows["original"]["source"] for rows in groups.values()]
    print(f"\n{label}: {len(groups)} source/walk units from {len(set(sources))} sources")
    print("metric        original   stretched   ablated   stretch-original [95% source CI]")
    for name in METRICS:
        orig = np.array([r["original"][name] for r in groups.values()])
        stretch = np.array([r["stretched"][name] for r in groups.values()])
        ablate = np.array([r["ablated"][name] for r in groups.values()])
        if len(groups) > 1:
            lo, hi = cluster_ci(stretch-orig, sources)
        else:
            lo, hi = float("nan"), float("nan")
        print(f"{name:<13} {orig.mean():8.3f} {stretch.mean():11.3f} {ablate.mean():9.3f} "
              f"{(stretch-orig).mean():+10.3f} [{lo:+.3f}, {hi:+.3f}]")


def show_contrast(label, treatment, baseline, treatment_name, baseline_name):
    pairs = sorted(set(treatment) & set(baseline))
    pairs = [pair for pair in pairs if set(CONDITIONS) <= set(treatment[pair])
             and set(CONDITIONS) <= set(baseline[pair])]
    if not pairs:
        raise ValueError(f"no complete source/walk pairs for {label}")
    mismatched_sources = [pair for pair in pairs
                          if treatment[pair]["original"]["source"] != baseline[pair]["original"]["source"]]
    if mismatched_sources:
        raise ValueError(f"source mismatch for pairs: {mismatched_sources}")
    sources = [treatment[pair]["original"]["source"] for pair in pairs]
    print(f"\n{label}: {treatment_name} - {baseline_name} on {len(pairs)} source/walk units "
          f"from {len(set(sources))} sources")
    print("metric        original   stretched   ablated   DiD contrast [95% source CI]   "
          "walk+ source+ within-source SD")
    for name in METRICS:
        gain = {condition: np.array([
            treatment[pair][condition][name] - baseline[pair][condition][name] for pair in pairs
        ]) for condition in CONDITIONS}
        did = gain["stretched"] - gain["original"]
        if len(pairs) > 1:
            lo, hi = cluster_ci(did, sources)
        else:
            lo, hi = float("nan"), float("nan")
        by_source = defaultdict(list)
        for source, value in zip(sources, did):
            by_source[source].append(value)
        source_means = np.array([np.mean(values) for values in by_source.values()])
        within_sd = np.mean([np.std(values) for values in by_source.values()])
        print(f"{name:<13} {gain['original'].mean():8.3f} {gain['stretched'].mean():11.3f} "
              f"{gain['ablated'].mean():9.3f} {did.mean():+8.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"{np.mean(did > 0):6.1%} {np.mean(source_means > 0):7.1%} {within_sd:16.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-data", required=True)
    ap.add_argument("--matched-data", required=True)
    ap.add_argument("--old", nargs="*")
    ap.add_argument("--matched", nargs="*")
    ap.add_argument("--no-demo", nargs="*")
    ap.add_argument("--no-demo-data")
    ap.add_argument("--sft-context", nargs="*")
    ap.add_argument("--sft-data")
    ap.add_argument("--json-output", help="safe per-source metrics without paths, captions, or generations")
    args = ap.parse_args()
    exported = []
    matched_by_model = {}
    no_demo_by_model = {}
    sft_by_model = {}

    def collect(label, data, output, matched):
        groups, model = metrics(data, output, matched)
        show(f"{label} {output}", groups)
        for pair, conditions in sorted(groups.items()):
            for condition, values in sorted(conditions.items()):
                exported.append({
                    "setting": label, "model": Path(output).name,
                    "base_model": model, "source": values["source"], "pair": pair,
                    "seed": values["seed"], "walk_seed_index": values["walk_seed_index"],
                    "condition": condition, **{key: float(values[key]) for key in METRICS},
                })
        return model, groups

    for path in args.old or []:
        collect("single_model", args.old_data, path, False)
    for path in args.matched or []:
        model, groups = collect("matched_G_to_G_minus_M", args.matched_data, path, True)
        matched_by_model[model] = groups
    for path in args.no_demo or []:
        if not args.no_demo_data:
            ap.error("--no-demo-data is required with --no-demo")
        model, groups = collect("no_demo", args.no_demo_data, path, True)
        no_demo_by_model[model] = groups
    for path in args.sft_context or []:
        if not args.sft_data:
            ap.error("--sft-data is required with --sft-context")
        model, groups = collect("sft_caption_plus_partial", args.sft_data, path, True)
        sft_by_model[model] = groups
    for model in sorted(set(matched_by_model) & set(no_demo_by_model)):
        show_contrast(f"Double-copy ICL gain {model}", matched_by_model[model], no_demo_by_model[model],
                      "double-copy", "single-copy")
    for model in sorted(set(sft_by_model) & set(no_demo_by_model)):
        show_contrast(f"SFT context gain {model}", sft_by_model[model], no_demo_by_model[model],
                      "SFT caption+partial", "single-copy")
    for model in sorted(set(matched_by_model) & set(sft_by_model)):
        show_contrast(f"Double-copy vs SFT {model}", matched_by_model[model], sft_by_model[model],
                      "double-copy", "SFT caption+partial")
    if args.json_output:
        dest = Path(args.json_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(exported)} path-free metric rows to {dest}")


if __name__ == "__main__":
    main()
