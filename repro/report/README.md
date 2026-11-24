# Report statistics reproduction

This directory regenerates the final report's **quantitative** tables and plots from a frozen, geometry-free evidence snapshot. Run from this directory:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

Outputs are written to `tables/`, `figures/`, and `generated/`. No GPU, network request, BrickNet dataset, checkpoint, or credential is needed. The code was checked with Python 3.11. The main table and summary CSV match the verified report bundle byte for byte; the stopping table has the same values after line-ending normalization. Numerical confidence intervals are regenerated from the observations.

## Snapshot contents

The Frontier-order Results table uses data/frontier_100.json. It contains the same 100 pseudonymous sources as the DFS file, with per-draw Pose F1, Exact@4 flags, and proposal log-odds for the two compared models. The DFS file now also retains Exact@4 flags.

- `data/completion_100.json`: 15 checkpoint rows with four sampled completion scores for each condition and each of 100 pseudonymous sources. It contains numeric scores and flags only, not prompts, generated text, graphs, captions, or poses.
- `data/completion_18.json`: historical 18-source aggregate scores, kept separate from the New100 benchmark.
- `data/physical.json`: numeric independent-generation lengths for natural and EOS-suppressed decoding, 64 draws per checkpoint.
- `data/training.json` and `data/grpo_training.json`: numeric training and development curves needed for the report plots; local file paths, configuration payloads, and model artifacts have been removed.
- `data/dataset_summary.json`: context lengths and conditions without source identities or structures.
- `data/proposal_dlo_100/`: teacher-forced, complete-target log-odds scores and 100,000-resample summary, with source names replaced by pseudonyms. The report's 12 current PT rows use this DLO definition; historical rows lacking it remain blank.
- `data/smoke_example.jsonl`: a separately authored toy assembly for parser and loss-mask smoke checks. It is not part of any reported score.

The raw BrickNet validation structures, 100-case prompts, target paths, generated assemblies, LDraw renders, and captions are omitted because permission to redistribute the gated dataset and its derived geometry has not been established. After receiving access from the [BrickNet authors](../../baseline/bricknet/BrickNet/DATA.md), the frozen construction and inference scripts under `repro/evaluation/` document the New100 procedure. The original cluster run paths and checkpoint identities are described in the report's internal provenance records but are not required for numeric replotting.

## Figures and limits

The paper's two-panel ICL-IWL scatter is regenerated as figures/icl_iwl_scatter.pdf. The three numerical tables embedded directly in the Results section are generated as tables/manuscript_objective.tex, tables/manuscript_frontier.tex, and tables/manuscript_tradeoff.tex.

The three numerical tables embedded directly in the paper's Results section are regenerated as tables/manuscript_objective.tex, tables/manuscript_frontier.tex, and tables/manuscript_tradeoff.tex. Their rounded entries were checked against the paper.

`scripts/reproduce.py` generates every numeric plot and table in the report. `scripts/compose_figures.py` contains the composition code for the qualitative image panels and method diagram, but the original input renders are withheld; those panels require authorized data and model outputs. Source-specific confidence intervals quantify variation across fixed test cases, not across training seeds. The tracked `RL/step9_grpo/grpo.py` and `RL/common/rewards.py` match the hashes recorded for the reported v2.1 run.
