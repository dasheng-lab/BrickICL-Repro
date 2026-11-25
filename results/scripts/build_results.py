"""Build the project's reported numerical tables and plots from public observations."""
import json
from pathlib import Path
import runpy
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
runpy.run_path(str(root / "scripts/reproduce_manuscript_tables.py"), run_name="__main__")

physical = json.loads((root / "data/physical.json").read_text(encoding="utf-8"))
order = [
    ("B0", "frozen_pt"), ("B-M", "post_pt"), ("B-G", "fullg_pt"),
    ("B-DFS", "plain"), ("B-LATE", "masked"), ("Q-M", "direct_m"),
    ("Q-G", "direct_g"), ("Q-DFS", "qwen_paired_dfs"),
    ("Q-LATE", "qwen_paired_last"), ("Q-G-R1", "regularized_g"),
    ("Q-G-R2", "regularized_g_wd010_do015"), ("B-RL", "grpo_pt"),
]
lines = [
    r"\begin{table}[ht]\centering\small",
    r"\caption{Both stopping policies, 64 draws each. All lengths count bricks and are capped at 100. No-EOS is a separately sampled intervention, not deletion of EOS from a saved output.}",
    r"\label{tab:stopping}",
    r"\begin{tabular}{@{}lrrrr@{}}\toprule",
    r"ID & $V$ & $C$ & $V_{\neg\mathrm{EOS}}$ & $C_{\neg\mathrm{EOS}}$ \\ \midrule",
]
for label, arm in order:
    values = [
        mean(float(row[metric]) for row in physical[arm][mode])
        for mode in ("natural", "no_eos")
        for metric in ("valid_length", "collision_free_length")
    ]
    lines.append(label + " & " + " & ".join(f"{value:.2f}" for value in values) + r" \\")
lines.append(r"\bottomrule\end{tabular}\end{table}")
(root / "tables/stopping.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

training = json.loads((root / "data/training.json").read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
for arm, label, color in (
    ("direct_g", "Q-G", "#2b6f99"),
    ("regularized_g", "Q-G-R1", "#d47a2f"),
    ("regularized_g_wd010_do015", "Q-G-R2", "#157f79"),
):
    logs = training[arm]["logs"]
    train = [row for row in logs if row.get("event") == "train"]
    values = np.array([row["loss"] for row in train])
    axes[0].plot(
        [row["optimizer_step"] for row in train],
        [values[max(0, i - 49):i + 1].mean() for i in range(len(values))],
        label=label, color=color, lw=1.3,
    )
    dev = [row for row in logs if row.get("event") == "dev" and row["optimizer_step"] >= 250]
    axes[1].plot(
        [row["optimizer_step"] for row in dev],
        [row["dev"]["mean"] for row in dev],
        "o-", label=label, color=color, lw=1, ms=3,
    )
for ax, title in zip(axes, ("Training CE: 50-step mean", "Validation CE: steps 250 onward")):
    ax.set(title=title, xlabel="Optimizer step", ylabel="Cross-entropy")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_xlim(250, 2250)
axes[0].set_ylim(.15, .40)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(root / "figures" / f"regularization_curves.{ext}", dpi=200, bbox_inches="tight", pad_inches=.04)
plt.close(fig)
print("Built the three main tables, ICL-IWL scatter, regularization plot, and stopping table.")
