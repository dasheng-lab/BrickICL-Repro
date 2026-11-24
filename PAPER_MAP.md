# Map from the final report to this repository

The teacher-forced Proposal DLO scoring implementation is preserved as source only at repro/cluster/proposal_dlo_100/; the anonymous scores used for all numeric tables are at repro/report/data/proposal_dlo_100/.

The paths below follow the final report's Methods, Results, Discussion, and Appendix structure. Run the quantitative reproduction from the repository root:

    cd repro/report
    python -m pip install -r requirements.txt
    python scripts/reproduce.py

The command writes plots to figures/, numerical tables to tables/, and supporting summaries to generated/. It does not need the restricted BrickNet dataset or a GPU.

| Paper location | Reproduction material |
|---|---|
| Methods: repeated-module augmentation | brickicl/posttrain/build_dataset.py, build_v2.py, build_v3.py, and the source-only structural run under repro/cluster/runs/structural_20261114/ |
| Methods: substructure-last ordering | brickicl/posttrain/reference_last.py and materialize_reference_last.py; paired DFS/last source in nzd-supertry/step3_data/ and repro/cluster/runs/qwen3_nzd_v2_20261120/ |
| Methods: M-only and Full-G supervision | brickicl/posttrain/data.py, train_bricknet06.py, train_fullg06.py; raw-Qwen and regularization run snapshots under repro/cluster/runs/ |
| Methods: GRPO v2.1 | RL/step9_grpo/grpo.py, RL/common/rewards.py, and repro/cluster/runs/grpo_v21_eval_20261122/ |
| Results: BrickNet objective table | repro/report/tables/manuscript_objective.tex, generated from data/completion_100.json and data/proposal_dlo_100/ |
| Results: DFS versus Frontier table | repro/report/tables/manuscript_frontier.tex, generated from data/completion_100.json, data/frontier_100.json, and proposal log-odds; source-only Frontier run in repro/cluster/frontier_testset/ |
| Results: ICL-IWL trade-off table and scatter | repro/report/tables/manuscript_tradeoff.tex and figures/icl_iwl_scatter.pdf, generated from the same completion data and data/physical.json |
| Results: regularization curves | repro/report/figures/regularization_curves.pdf from data/training.json |
| Results: GRPO curves and stopping comparison | repro/report/figures/rl_curves.pdf, figures/eos_effect.pdf, and tables/stopping.tex from data/grpo_training.json and data/physical.json |
| Appendix: uncertainty and older suite | repro/report/tables/confidence_intervals.tex, tables/all_suites.tex, and tables/sft.tex |

The report's method diagrams and qualitative assembly panels require original renders or derived geometry. Those inputs are not redistributed because BrickNet validation data is gated and permission to redistribute its derived geometries has not been established. The composition code is in repro/report/scripts/compose_figures.py, and authorized inference source is under repro/report/repro/evaluation/ and repro/cluster/.

RL/ contains only the reported GRPO work and its direct helpers. Earlier numbered RFT and DPO planning folders are intentionally omitted because they were not implemented as reported experiments.
