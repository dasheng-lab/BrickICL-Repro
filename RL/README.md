# GRPO experiment

The paper's reinforcement learning experiment is GRPO v2.1. The implementation and historical run notes are in step9_grpo/. Earlier numbered RFT and DPO planning stages were not implemented as reported experiments and are omitted from this public release.

The BrickNet GRPO checkpoint was trained from the component-last Full-G parent with a pose F2 reward. The exact training driver is step9_grpo/grpo.py (SHA-256 23e4388efe0b9f7bb3f057793166176524e14cb0f09a8202fd176c63e0363efc); the reward implementation is common/rewards.py (SHA-256 d70d0f73999f45ffc42ccecd2287ecaa29484ef6240ced19af643ff9ae616760). These hashes match the archived v2.1 training record.

From an authorized Linux GPU environment, the historical entry point is:

    cd RL
    ROUND=v2.1 PRESET=v2.1 bash step9_grpo/run_grpo.sh

The shell script expects authorized BrickNet-derived training prompts, the parent checkpoint, CUDA/vLLM dependencies, and host-specific paths configured through env.sh. The reward and sampling helpers also use nzd-supertry/step11_report_metrics/ and nzd-supertry/step5_eval/. This public release includes those source dependencies; it does not include gated inputs, checkpoints, or raw model outputs.

GRPO evaluation source is archived at repro/cluster/runs/grpo_v21_eval_20261122/. Geometry-free reward and evaluation curves are in repro/report/data/grpo_training.json, and the report's plots and tables regenerate with python repro/report/scripts/reproduce.py.
