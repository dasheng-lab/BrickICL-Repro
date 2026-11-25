"""Rebuild the three numeric tables embedded in the paper's Results section."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results"
DATA = ROOT / "data"
TABLES = ROOT / "tables"
ROW_END = " " + "\\" * 2


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def scored(cases, condition):
    rows = [r for r in cases if r["condition"] == condition]
    assert len(rows) == 100 and all(len(r["samples"]) == 4 for r in rows)
    f1 = sum(s["pose_f1"] for r in rows for s in r["samples"]) / 400
    exact4 = sum(any(s["exact"] for s in r["samples"]) for r in rows)
    return f1, exact4


def dlo(rows):
    values = {}
    for row in rows:
        values.setdefault(row["source_key"], {})[row["condition"]] = float(row["log_odds"])
    assert len(values) == 100 and all(set(v) == {"original", "stretched"} for v in values.values())
    return sum(v["stretched"] - v["original"] for v in values.values()) / 100


def dfs_dlo(arm):
    path = DATA / "reference_sensitivity" / "scores" / f"{arm}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [dict(source_key=r["source"], condition=r["condition"], log_odds=r["log_odds"]) for r in csv.DictReader(stream)]
    return dlo(rows)


def make_row(label, cases, delta):
    normal, normal_exact = scored(cases, "original")
    long, long_exact = scored(cases, "stretched")
    return f"{label} & {delta:.2f} & {normal:.3f} & {long:.3f} & {normal_exact} & {long_exact}" + ROW_END


def write(name, caption, heading, columns, rows):
    lines = [
        r"\begin{table}[htbp]", r"\centering\small", f"\\caption{{{caption}}}",
        r"\begin{tabular}{@{}" + columns + r"@{}}\toprule",
        heading + ROW_END, r"\midrule", *rows, r"\bottomrule\end{tabular}", r"\end{table}",
    ]
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


dfs = load("completion_100.json")
frontier = load("frontier_100.json")
physical = load("physical.json")

write(
    "manuscript_objective.tex",
    "Effect of the training objective on BrickNet (100 DFS sources).",
    r"Model & $\Delta$ log-odds & F1 original & F1 stretched & Exact@4 original (\%) & Exact@4 stretched (\%)",
    "lrrrrr",
    [make_row(label, dfs[arm]["cases"], dfs_dlo(arm)) for label, arm in
     [("Released BrickNet", "frozen_pt"), ("Full-G", "fullg_pt"), ("M-only", "post_pt")]],
)

frontier_rows = [
    make_row(
        f"{layout} / {label}",
        dfs[arm]["cases"] if layout == "DFS" else frontier["models"][arm]["cases"],
        dfs_dlo(arm) if layout == "DFS" else dlo(frontier["models"][arm]["proposal_log_odds"]),
    )
    for layout in ("DFS", "Frontier")
    for label, arm in (("Plain", "plain"), ("Component-last", "masked"))
]
write(
    "manuscript_frontier.tex",
    "BrickNet Full-G on the same sources under DFS and frontier orderings.",
    r"Test order / Training & $\Delta$ log-odds & F1 original & F1 stretched & Exact@4 original (\%) & Exact@4 stretched (\%)",
    "lrrrrr",
    frontier_rows,
)

tradeoff_rows = []
for label, arm in [
    ("BrickNet / Full-G", "fullg_pt"), ("BrickNet / M-only", "post_pt"),
    ("Qwen / Full-G", "direct_g"), ("Qwen / M-only", "direct_m"),
    ("Qwen / WD 0.01, DO 0.05", "direct_g"),
    ("Qwen / WD 0.05, DO 0.10", "regularized_g"),
    ("Qwen / WD 0.10, DO 0.15", "regularized_g_wd010_do015"),
]:
    valid = sum(float(r["valid_length"]) for r in physical[arm]["no_eos"]) / len(physical[arm]["no_eos"])
    tradeoff_rows.append(f"{label} & {dfs_dlo(arm):.2f} & {valid:.2f}" + ROW_END)
write(
    "manuscript_tradeoff.tex",
    "Reference sensitivity and independent generation.",
    r"Initialization / Objective & $\Delta$ log-odds & NoEOSValid (bricks)",
    "lrr",
    tradeoff_rows,
)
print("Rebuilt three Results-section tables from per-case observations.")

# The Results-section ICL-IWL scatter uses the same per-case DLO and 64-draw
# independent-generation observations as the trade-off table.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def valid_length(arm):
    rows = physical[arm]["no_eos"]
    return sum(float(r["valid_length"]) for r in rows) / len(rows)


fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})
left = [
    (1, "frozen_pt", "#276ab1"), (2, "post_pt", "#276ab1"),
    (3, "fullg_pt", "#276ab1"), (4, "plain", "#276ab1"),
    (5, "direct_m", "#df7a1d"), (6, "direct_g", "#df7a1d"),
    (7, "qwen_paired_dfs", "#df7a1d"),
]
for number, arm, color in left:
    x, y = valid_length(arm), dfs_dlo(arm)
    axes[0].scatter(x, y, s=330, color=color, zorder=3)
    axes[0].text(x, y, str(number), ha="center", va="center", color="white", weight="bold", fontsize=9, zorder=4)
right = [
    (6, "direct_g", "#687381"),
    (8, "regularized_g", "#303846"),
    (9, "regularized_g_wd010_do015", "#858e99"),
]
for number, arm, color in right:
    x, y = valid_length(arm), dfs_dlo(arm)
    axes[1].scatter(x, y, s=370, color=color, zorder=3)
    axes[1].text(x, y, str(number), ha="center", va="center", color="white", weight="bold", fontsize=9, zorder=4)
for ax, title in zip(axes, ["(a) Training objective and baselines", "(b) Regularization zoom"]):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("IWL: no-EOS valid length (bricks)")
    ax.grid(alpha=.18)
axes[0].set_ylabel("ICL: proposal Δ log-odds")
axes[1].set_ylabel("ICL: proposal Δ log-odds")
axes[0].set_xlim(-5, 105)
axes[0].set_ylim(0, 70)
axes[1].set_xlim(48, 66)
axes[1].set_ylim(19.5, 24)
fig.suptitle("ICL and IWL under different training choices", fontsize=13, weight="bold")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ROOT / "figures" / f"icl_iwl_scatter.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Rebuilt Results-section ICL-IWL scatter.")
