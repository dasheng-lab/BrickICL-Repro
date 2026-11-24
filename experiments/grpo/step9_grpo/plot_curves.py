"""Curves of Δlog-odds and Pose F1 against the training step, one line per arm. (CPU, matplotlib; Chinese labels
need a CJK font: Microsoft YaHei / SimHei / Noto Sans CJK)

Input: a JSON file with "steps" and, per arm, "dlogodds" ([mean, lo, hi] per step) and the pose_f1.py columns
(pose_f1_sunshq_stretched, pose_f1_sunshq_dev, ...), as step9_grpo/figures/v2.1_curves.json.

Usage: python step9_grpo/plot_curves.py step9_grpo/figures/v2.1_curves.json --out step9_grpo/figures/v2.1_curves.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order (dataviz palette)
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_xlabel("训练步", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.data, encoding="utf-8"))
    steps, arms = d["steps"], d["arms"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=160)
    panels = [
        ("Δlog-odds（加长参照开发集，187 对）", "Δlog-odds", "dlogodds", "{:+.2f}"),
        ("Pose F1，参照加长（加长参照开发集，sunshq 算法）", "Pose F1", "pose_f1_sunshq_stretched", "{:.3f}"),
        ("Pose F1（开发集 477 题，sunshq 算法）", "Pose F1", "pose_f1_sunshq_dev", "{:.3f}"),
    ]
    for ax, (title, ylabel, key, fmt) in zip(axes, panels):
        for i, (name, a) in enumerate(arms.items()):
            c = SERIES[i]
            if key == "dlogodds":
                y = [v[0] for v in a[key]]
                ax.fill_between(steps, [v[1] for v in a[key]], [v[2] for v in a[key]], color=c, alpha=0.13, linewidth=0)
            else:
                y = a[key]
            ax.plot(steps, y, color=c, linewidth=2, marker="o", markersize=4.5, markeredgecolor="white",
                    markeredgewidth=1.2, label=name, zorder=3)
            end_label(ax, steps[-1], y[-1], f"{name} {fmt.format(y[-1])}")
        style(ax, title, ylabel)
        ax.set_xticks(steps)
        ax.set_xlim(-10, steps[-1] + 85)
    axes[0].annotate("阴影：95% 区间（按对重采样）", (0.02, 0.03), xycoords="axes fraction", fontsize=8, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(l), frameon=False, fontsize=9.5, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("GRPO r1 与 v2.1：给第一个零件时随训练步的变化", x=0.01, ha="left", fontsize=12.5, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
