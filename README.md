# BrickICL: in-context learning in LEGO assembly

For a section-by-section guide to the final paper's methods, figures, and tables, see [PAPER_MAP.md](PAPER_MAP.md).

CS182 project by Sixian Shen, Zidong Ning, Shengqing Sun, and Junyang Wu. [简体中文](README.zh-CN.md)

We study whether 0.6B assembly models use a visible repeated subassembly to complete another copy, and how this relates to generation without a LEGO reference. The fixed evaluation compares original and lengthened references on 100 sources from BrickNet's validation split. The main report compares 12 current post-training recipes; a historical 18-source suite and three earlier checkpoints are kept separate.

## Reproduce the report's numerical results

The repository contains **de-identified, per-case numeric observations** and code for the report's quantitative plots and tables. No BrickNet graph, caption, path text, LDraw scene, or rendered validation model is included.

```bash
cd repro/report
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

This regenerates the numerical tables in `tables/`, plots in `figures/`, and bootstrap summaries in `generated/` without a GPU, checkpoint, or dataset download. The bootstrap samples whole source cases for paired completion results and independent draws for reference-free generation. `repro/report/README.md` lists the input snapshots, generated outputs, and verification commands. The paper's qualitative renders require authorized BrickNet data and are not part of the public numeric snapshot.

## Data access and small test set

BrickNet's full training and validation datasets are **not public downloads**. Request access using the [form linked by the BrickNet authors](https://forms.gle/dm4eYSa5gh4DqzRT6), as described in [BrickNet's DATA.md](baseline/bricknet/BrickNet/DATA.md). After receiving an authorized share URL, set `BRICKNET_SHARE_URL` locally and run `baseline/repro/fetch_data.sh`. Do not commit the URL, downloaded data, or derived geometry. The 100-source benchmark inputs are likewise omitted pending permission to redistribute them.

`repro/report/data/smoke_example.jsonl` is a **separately authored toy assembly**, not a BrickNet validation case or a source of reported scores. With BrickNet and its public collision meshes installed, run `python repro/report/scripts/smoke_example.py` to check parsing, connectivity, and collision scoring. Passing `--tokenizer MODEL_OR_PATH` also checks the completion loss mask. This small fixture verifies the input format; numerical report reproduction uses the de-identified observations above.

## Training and evaluation code

- `brickicl/posttrain/` contains repeated-module construction, full-assembly and missing-module objectives, LoRA training, and the associated evaluation utilities.
- `brickicl/sampling/` and `brickicl/eval/` contain the assembly samplers and earlier controlled probes.
- `repro/report/scripts/` contains the frozen report analysis. `repro/report/repro/evaluation/` contains the New100 input-construction and scoring source snapshot; running it on the original cases requires authorized BrickNet data and model weights.
- `repro/cluster/runs/` contains source-only snapshots found in the cluster audit, including the DFS corpus, 100-source benchmark, GRPO evaluation, no-EOS generation, Qwen controls, and regularization runs. See its [audit](repro/cluster/CLUSTER_AUDIT.md).
- `RL/step9_grpo/` contains the GRPO v2.1 training driver; RL/common/rewards.py contains the reward implementation used for B-RL.
- `baseline/bricknet/BrickNet/` contains a pinned upstream BrickNet source copy. Its `DATA.md` explains the dataset request route.

The report's B-RL results came from a GRPO v2.1 run. The tracked `RL/step9_grpo/grpo.py` and `RL/common/rewards.py` match the SHA-256 hashes in the training record. Repeating that training still requires the authorized BrickNet-derived inputs, model weights, and a compatible GPU environment. The numeric report plots and tables regenerate without them.

## Scope of the evidence

The 100-source evaluation uses strict depth-first serialization. The older 18-source results use a different source set and are reported separately. Checkpoints were selected on development data. The public numeric observations contain no raw generated assembly text; they support checking the reported statistics, not rerunning model inference without the gated inputs. No multi-seed training or physical-stability evaluation is claimed.

The original BrickNet authors retain their own licensing and access terms. This repository does not redistribute their gated dataset or private share URL.
