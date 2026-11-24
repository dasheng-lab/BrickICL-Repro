# BrickICL: in-context learning in LEGO assembly

CS182 project by Sixian Shen, Zidong Ning, Shengqing Sun, and Junyang Wu.

We study whether a 0.6B assembly model can use a visible repeated subassembly to complete another copy, and how that ability relates to independent LEGO generation. The final report compares repeated-module augmentation (RMA), substructure-last augmentation (SLA), missing-module supervision (M-only), full-assembly supervision (Full-G), regularization, and GRPO. The main completion evaluation uses 100 paired sources from BrickNet's gated validation split.

## Reproduce the report's quantitative results

From the repository root:

```bash
python -m pip install -r reproduction/requirements.txt
python reproduction/scripts/reproduce.py
```

This regenerates the numeric tables in `reproduction/tables/`, plots in `reproduction/figures/`, and supporting summaries in `reproduction/generated/`. It needs Python 3.11 and the listed packages, but no GPU, model checkpoint, network request, or BrickNet download. The checked-in `reproduction/data/` contains pseudonymous per-case numeric observations and training curves. It contains no validation graphs, captions, assembly paths, generated assembly text, LDraw scenes, or rendered geometry. Bootstrap intervals over the completion benchmark resample whole source cases; they do not estimate variation across training seeds.

The manuscript's three Results tables are `reproduction/tables/manuscript_objective.tex`, `manuscript_frontier.tex`, and `manuscript_tradeoff.tex`. The ICL–IWL scatter is `reproduction/figures/icl_iwl_scatter.pdf`. The historical 18-source suite in `reproduction/data/completion_18.json` is kept separate from the current 100-source benchmark.

## Paper-to-code guide

| Final report section | Implementation and reproducible output |
| --- | --- |
| Methods — RMA data curation | `brickicl/posttrain/build_dataset.py`, `build_v2.py`, `build_v3.py`; construction and training snapshots in `experiments/augmentation/` |
| Methods — SLA data curation and DFS/LATE comparison | `brickicl/posttrain/reference_last.py` and `materialize_reference_last.py`; paired assembly-order pipeline in `experiments/assembly_order/`; Qwen paired-order training in `experiments/supervised_training/qwen3_paired_order_20261120/` |
| Methods — M-only and Full-G objectives | `brickicl/posttrain/data.py`, `train_bricknet06.py`, `train_fullg06.py`; Qwen initialization and regularization runs in `experiments/supervised_training/` |
| Methods — GRPO | `experiments/grpo/step9_grpo/grpo.py` and `experiments/grpo/common/rewards.py`; evaluation source in `evaluation/snapshots/grpo_v21_eval_20261122/` |
| Results — completion and objective comparison | `evaluation/new100/source/` for benchmark construction and scoring; `reproduction/data/completion_100.json` and `data/proposal_dlo_100/` for anonymized scores; `reproduction/tables/manuscript_objective.tex` |
| Results — DFS versus Frontier ordering | `evaluation/frontier/source/`; `reproduction/data/frontier_100.json` and `tables/manuscript_frontier.tex` |
| Results — ICL–IWL trade-off | `reproduction/data/completion_100.json`, `data/physical.json`, `tables/manuscript_tradeoff.tex`, and `figures/icl_iwl_scatter.pdf` |
| Results — regularization, GRPO, and stopping | `reproduction/data/training.json`, `data/grpo_training.json`, `figures/regularization_curves.pdf`, `figures/rl_curves.pdf`, `figures/eos_effect.pdf`, and `tables/stopping.tex` |
| Appendix — uncertainty and additional suites | `reproduction/tables/confidence_intervals.tex`, `all_suites.tex`, and `sft.tex` |

`brickicl/` is the importable project package. `experiments/` groups data construction and training by the paper's interventions; `evaluation/` contains benchmark and scoring implementations; `reproduction/` contains the public numeric evidence and reporting scripts; `analysis/snapshots/` preserves report assembly sources; and `baseline/bricknet/BrickNet/` is the pinned upstream BrickNet source copy. Timestamped run directories under `experiments/`, `evaluation/snapshots/`, and `analysis/snapshots/` are source-only historical snapshots. Their original launch scripts record cluster-specific paths and require adaptation, authorized data, weights, and a compatible environment before a fresh model run.

The main Proposal DLO implementation is `evaluation/proposal_dlo/source/`. Its public pseudonymous scores are in `reproduction/data/proposal_dlo_100/`. The report's qualitative panels and method diagram need original renders or derived geometry; `reproduction/scripts/compose_figures.py` preserves the composition code, but those inputs are not redistributed.

## BrickNet data and a small public test

BrickNet's full training and validation datasets are not public downloads. Request access through the [form given by the BrickNet authors](https://forms.gle/dm4eYSa5gh4DqzRT6), also linked in [BrickNet's data instructions](baseline/bricknet/BrickNet/DATA.md). After access is granted, set `BRICKNET_SHARE_URL` locally and run `baseline/repro/fetch_data.sh`. Do not commit the share URL, downloaded dataset, or validation-derived geometry. The original 100-case benchmark prompts and targets are omitted because permission to redistribute the gated inputs or their derived geometry has not been established.

The independent [toy test fixture](reproduction/data/smoke_example.jsonl) is also stored in [Google Drive](https://drive.google.com/file/d/16NDCaDN4kLeSpejVeYvroTZIK34lPJxa/view?usp=drivesdk). The Drive copy is readable within the UC Berkeley domain; the repository copy is public to everyone. This example was authored separately from BrickNet validation data and is not used in any reported score. With BrickNet and its public collision meshes installed, run:

```bash
python reproduction/scripts/smoke_example.py
```

This checks parsing, connectivity, and collision scoring. Add `--tokenizer MODEL_OR_PATH` to check the completion loss mask. The fixture tests the code path; the numeric report reproduction above uses the public de-identified observations.

## Training provenance and scope

The reported B-RL run is GRPO v2.1. SHA-256 of `experiments/grpo/step9_grpo/grpo.py` is `23e4388efe0b9f7bb3f057793166176524e14cb0f09a8202fd176c63e0363efc`; SHA-256 of `experiments/grpo/common/rewards.py` is `d70d0f73999f45ffc42ccecd2287ecaa29484ef6240ced19af643ff9ae616760`. These match the recorded training sources. The preserved GRPO code refers to historical assembly-order paths; `experiments/grpo/prepare_layout.py` creates ignored compatibility links when `experiments/grpo/step9_grpo/run_grpo.sh` starts, leaving those hash-verified files unchanged. Earlier numbered RL plans are absent because they were not implemented as reported experiments.

The cluster audit recovered source-only snapshots for RMA/DFS preparation, the New100 and Frontier evaluations, Proposal DLO scoring, Qwen controls and regularization, GRPO evaluation, no-EOS generation, and report analysis. The repository does not include run plans, original prompts or outputs, checkpoints, or restricted data. The public observations support reproducing the paper's numerical plots and tables. Rerunning inference or training needs authorized inputs and model weights. The report does not claim multi-seed training or physical-stability evaluation.

The original BrickNet authors retain their own licensing and data-access terms.
