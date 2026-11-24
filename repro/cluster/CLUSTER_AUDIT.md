# Cluster source audit, 2026-11-24

A second scan found the paper's Frontier-order experiment under /home/shensx3017/brickicl_eval/frontier_testset. Its 14 source files are under frontier_testset/. Per-case numeric observations were pseudonymized into ../report/data/frontier_100.json. Raw inputs, targets, generated text, and model outputs are excluded.

The same account also held the main Proposal DLO scorer. Its source-only evaluator and summarizer are under proposal_dlo_100/. The published numeric CSVs are under ../report/data/proposal_dlo_100/, with source IDs replaced by case aliases.

The primary project was scanned at `/home/sunshq114514/brickICL`. The report's source paths were compared with the local Git checkout and the report bundle. The original cluster tree was not changed for this release. The scanned run directories below had source-only `.py` and `.sh` files copied under `runs/` here, in addition to the five Qwen training run snapshots described in the parent README.

| Run directory | Source files | Report role |
|---|---:|---|
| `dfs_posttrain_20261114` | 25 | Strict DFS sampling and corpus preparation used by New100 construction |
| `dfs_70k_20261114` | 15 | Earlier DFS training recipe and control |
| `structural_20261114` | 19 | Structural data construction and training |
| `structural_eval_20261115` | 17 | Original completion scoring implementation |
| `eval100_20261122` | 20 | 100-source benchmark construction and main inference |
| `grpo_v21_eval_20261122` | 8 | GRPO v2.1 evaluation |
| `no_eos_eval_20261120` | 11 | Reference-free generation with EOS suppressed |
| `context_length_distribution_20261122` | 2 | Context-length figure source |
| `latest_results_20261122` | 16 | Result collection and publication source |
| `final_report_20261123` | 14 | Report analysis and figure source |
| `final_report_updated_20261123` | 2 | Final report refresh audit |
| `qwen3_direct_eval_20261119` | 37 | Raw-Qwen control evaluation and service source |
| `qwen3_fullg_reg_eval_20261119` | 6 | Moderate-regularization evaluation |
| `qwen3_fullg_reg_wd010_do015_eval_20261121` | 10 | Stronger-regularization evaluation |
| `qwen3_nzd_v2_plain_eval_20261120` | 10 | DFS paired-order evaluation |
| `qwen3_nzd_v2_masked_eval_20261120` | 10 | Substructure-last paired-order evaluation |

The archive selected 222 source files. Duplicate copies of playground services and the report's `author_bundle` are omitted from the public export; canonical playground source is under `brickicl/`, and the sanitized report analysis is under `repro/report/`. The historical GRPO training driver is the tracked `RL/step9_grpo/grpo.py` with SHA-256 `23e4388efe0b9f7bb3f057793166176524e14cb0f09a8202fd176c63e0363efc`, matching the training record. A newer file in a teammate's current cluster home is not presented as the historical run.

The `runs/` snapshots contain code only. Original run plans, model weights, generated assembly text, BrickNet validation graphs, LDraw scenes, screenshots, and rendered geometry are excluded. Report plots and tables instead regenerate from the geometry-free observations in `../report/data/`. Because the original benchmark inputs cannot be redistributed without permission, the historical inference scripts are provided for authorized reruns, not as a public turnkey benchmark.
