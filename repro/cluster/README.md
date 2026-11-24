# Cluster training source snapshots

The paper's Frontier-order comparison was found in a second cluster account and is archived as source-only code under frontier_testset/. Its public numeric observations are in ../report/data/frontier_100.json. Earlier RL step1–8 planning folders are omitted; the reported RL implementation is GRPO v2.1 under ../../RL/step9_grpo/.

These scripts are archived from the recorded project runs. They document the model recipes used in the report; they are **not** a second copy of the dataset or a turnkey GPU environment. The supported BrickNet-initialized M-only and Full-G data/training modules live in `../../brickicl/posttrain/`. The earlier paired-order implementation is in `../../nzd-supertry/`.

| Report recipe | Source snapshot | Role |
|---|---|---|
| Q-G and Q-M controls | `runs/qwen3_direct_pt_20261117/`, `runs/qwen3_three_job_queue_20261119/` | Raw-Qwen initialization, dataset and schedule preparation, supervised training |
| Q-DFS and Q-LATE | `runs/qwen3_nzd_v2_20261120/` | Paired order comparison on the same source assemblies |
| Q-G-R1 | `runs/qwen3_fullg_reg_20261119/` | Full-G with weight decay 0.05 and LoRA dropout 0.10 |
| Q-G-R2 | `runs/qwen3_fullg_reg_wd010_do015_20261121/` | Full-G with weight decay 0.10 and LoRA dropout 0.15 |
| B-RL | `../../RL/step9_grpo/` | GRPO v2.1 training driver from the repository, matching the recorded training hash |

The Qwen `train.py` files take a `--config` path. Their original configs refer to gated BrickNet-derived token caches, model files, and exact code/data hashes; those assets are not redistributed. Build new configs and caches only after obtaining authorized BrickNet data. The copied `launch.sh` scripts record the original cluster layout and contain host-specific paths; adapt the environment before running them elsewhere. Report evaluation construction and scoring code is in `../report/repro/evaluation/`; numerical plotting and tables are reproduced without GPUs by `../report/scripts/reproduce.py`.

## GRPO provenance

The v2.1 training record lists SHA-256 `23e4388efe0b9f7bb3f057793166176524e14cb0f09a8202fd176c63e0363efc` for `step9_grpo/grpo.py` and `d70d0f73999f45ffc42ccecd2287ecaa29484ef6240ced19af643ff9ae616760` for `common/rewards.py`. Both tracked files under `../../RL/` match these hashes. Rerunning the training still requires authorized inputs and a compatible GPU environment. The numeric report plots and tables regenerate from the archived observations without retraining.

## Additional cluster audit

The `runs/` directory also includes source-only snapshots for the DFS and structural training pipelines, the 100-source benchmark construction and evaluation, GRPO evaluation, no-EOS generation, context-length analysis, and final-report analysis. These were found by scanning the primary cluster project on 2026-11-24. The audit and exact directory list are in [CLUSTER_AUDIT.md](CLUSTER_AUDIT.md). Runtime plans, prompts, model outputs, checkpoints, and gated dataset files are deliberately absent. These scripts document the historical run; some expect the original cluster layout and need path/configuration changes before they can be rerun on authorized data.
